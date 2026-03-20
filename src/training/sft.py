"""
Cortical Creative Loop (CCL) Training

Training modes:
  - mode='ccl': Brain-inspired cortical loop (PERCEIVE→INTENT→PREDICT→ERROR→SELECT)
  - mode='simple': Basic SFT for comparison/fallback

The CCL format teaches the model to:
  1. PERCEIVE: Encode context (section, emotion, rhythm)
  2. INTENT: Form generation goals
  3. PREDICT: Generate draft output
  4. ERROR: Detect prediction errors (emotion mismatch, rhythm weakness, etc.)
  5. SELECT: Choose final output

Stage 1 + 2:
  - Stage 1: General music SFT on full annotated corpus
  - Stage 2: Per-genre LoRA adapters

Kaggle-optimized defaults:
  - TRAIN_SUBSET_ROWS: 3000 (fast), 10000 (meaningful), 20000 (heavy)
  - SAVE_STEPS: 200 (frequent checkpoints for interruption recovery)
  - MAX_LENGTH: 320 (fits CCL format without truncation)
  - MAX_PER_ARTIST: 3 (reduces overconcentration)
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm

from peft import prepare_model_for_kbit_training
from src.model.checkpoint_loader import load_for_training
from src.model.lyrics_model import apply_lora, LyricsModel
from src.model.phonetic_head import phonetic_head_loss


def train_sft(
    stage: int = 1,
    genre: Optional[str] = None,
    data_path: str = "data/raw/all_songs.jsonl",
    val_path: Optional[str] = None,
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    output_dir: str = "checkpoints",
    batch_size: int = 2,              # Kaggle-safe default
    grad_accum_steps: int = 8,
    max_length: int = 320,            # Fits CCL format
    epochs: int = 1,
    lr: float = 1e-4,
    lora_rank: int = 32,
    alpha: Optional[int] = None,
    use_4bit: bool = True,
    style_vec_dir: Optional[str] = None,  # Off by default
    phonetic_loss_weight: float = 0.1,
    viral_conditioning: Optional["np.ndarray"] = None,
    # New CCL options
    training_mode: str = "ccl",       # 'ccl' or 'simple'
    save_steps: int = 200,            # Save every N steps
    max_per_artist: int = 3,          # Cap songs per artist
    curriculum_order: bool = True,    # Short/simple examples first
):
    """
    Cortical Creative Loop training.

    Parameters
    ----------
    training_mode : str
        'ccl' for brain-inspired cortical loop format
        'simple' for basic SFT format
    save_steps : int
        Save checkpoint every N steps (for interrupt recovery)
    max_per_artist : int
        Cap songs per artist to reduce overconcentration
    curriculum_order : bool
        Sort examples by complexity (short/simple first)
    """
    from pathlib import Path as P

    def save_checkpoint(mdl, tok, path):
        """Save PEFT adapter + tokenizer to path."""
        P(path).mkdir(parents=True, exist_ok=True)
        # Get the underlying model (handle accelerator wrapping if any)
        unwrapped = accelerator.unwrap_model(mdl) if hasattr(accelerator, 'unwrap_model') else mdl
        # If it's a LyricsModel wrapper, get the base PEFT model
        if hasattr(unwrapped, 'base'):
            peft_model = unwrapped.base
        else:
            peft_model = unwrapped
        # Save PEFT adapter
        peft_model.save_pretrained(path)
        tok.save_pretrained(path)
        print(f"  [sft] Checkpoint saved → {path}")

    # Import the appropriate dataset module
    if training_mode == "ccl":
        from src.training.cortical_dataset import create_ccl_dataloaders
        create_dataloaders_fn = lambda *args, **kwargs: create_ccl_dataloaders(
            *args, mode="ccl", max_per_artist=max_per_artist, **kwargs
        )
        print(f"[sft] Training mode: CORTICAL CREATIVE LOOP (CCL)")
    else:
        from src.training.dataset import create_dataloaders
        create_dataloaders_fn = create_dataloaders
        print(f"[sft] Training mode: simple SFT")

    accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps)

    lora_alpha = alpha if alpha is not None else lora_rank * 2

    if stage == 2:
        assert genre is not None, "Stage 2 requires --genre"
        output_subdir = f"{output_dir}/genre_{genre}"
    else:
        output_subdir = f"{output_dir}/stage1_general"

    # Log viral conditioning if provided
    if viral_conditioning is not None and np.linalg.norm(viral_conditioning) > 0:
        print(f"  Viral DNA conditioning: norm={np.linalg.norm(viral_conditioning):.3f}, "
              f"top dims={viral_conditioning[:4].round(3)}")

    Path(output_subdir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base, tokenizer = load_for_training(base_model, use_4bit=use_4bit and device == "cuda")

    # prepare_model_for_kbit_training MUST run before apply_lora on 4-bit models.
    # On T4 (14.5GB) the model already occupies ~13.3GB, leaving barely 500MB.
    # The default call tries to upcast ALL LayerNorm/embedding params to float32
    # which spikes memory and OOMs.  We avoid the spike by:
    #   1. Aggressively freeing memory first
    #   2. Disabling gradient checkpointing inside prepare (we enable it after LoRA)
    #   3. Manually casting only the trainable LoRA params — not the whole model
    if use_4bit and device == "cuda":
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_before = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  [sft] GPU free before prepare: {free_before:.2f} GB")
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=False,   # avoid float32 upcast spike
        )
        gc.collect()
        torch.cuda.empty_cache()
        print("  [sft] prepare_model_for_kbit_training applied")
    else:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    base = apply_lora(base, rank=lora_rank, alpha=lora_alpha)

    # Enable gradient checkpointing AFTER LoRA injection.
    # On 4-bit models this is safe and avoids the memory spike of doing it earlier.
    if use_4bit and device == "cuda":
        base.enable_input_require_grads()
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    elif not (use_4bit and device == "cuda"):
        pass  # already enabled above for non-4bit path
    model = LyricsModel(base, d_model=base.config.hidden_size)

    quantized_kaggle_safe = use_4bit and device == "cuda"
    if quantized_kaggle_safe:
        # 4-bit models are already placed by transformers/bitsandbytes. Moving the
        # whole wrapper again through `accelerate.prepare(model, ...)` can duplicate
        # allocations and OOM small GPUs like Kaggle T4s.
        embed_layer = base.get_input_embeddings()
        embed_device = embed_layer.weight.device
        embed_dtype = embed_layer.weight.dtype
        model.style_projector = model.style_projector.to(device=embed_device, dtype=embed_dtype)
        model.phonetic_head = model.phonetic_head.to(device=embed_device, dtype=embed_dtype)
        print(f"  Training wrapper pinned to {embed_device} without extra model move")

    if accelerator.is_main_process:
        model.base.print_trainable_parameters()

    train_dl, val_dl = create_dataloaders_fn(
        data_path, val_path, tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_dl) // grad_accum_steps) * epochs
    warmup_steps = total_steps // 10
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if quantized_kaggle_safe:
        optimizer, train_dl, scheduler = accelerator.prepare(
            optimizer, train_dl, scheduler
        )
    else:
        model, optimizer, train_dl, scheduler = accelerator.prepare(
            model, optimizer, train_dl, scheduler
        )

    global_step = 0
    last_saved_epoch = 0
    interrupted = False

    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", disable=not accelerator.is_main_process)

            for step, batch in enumerate(pbar):
                # For the quantized Kaggle path the model is NOT prepared by accelerator,
                # so accelerator.accumulate(model) cannot track sync boundaries correctly.
                # Use a manual step-based accumulation check instead.
                is_accumulation_step = ((step + 1) % grad_accum_steps != 0) and (step + 1 < len(train_dl))

                # Inject viral DNA: add as a bias to the style vector
                style_vec = batch.get("style_vec")
                if viral_conditioning is not None and style_vec is not None:
                    viral_t = torch.tensor(viral_conditioning, dtype=style_vec.dtype, device=style_vec.device)
                    sv_dim = style_vec.shape[-1]
                    v_dim  = viral_t.shape[0]
                    if v_dim < sv_dim:
                        viral_t = torch.nn.functional.pad(viral_t, (0, sv_dim - v_dim))
                    else:
                        viral_t = viral_t[:sv_dim]
                    style_vec = style_vec + 0.1 * viral_t.unsqueeze(0)

                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    phoneme_ids=batch.get("phoneme_ids"),
                    style_vec=style_vec,
                    labels=batch["labels"],
                )

                loss = out["lm_loss"]

                # Guard against NaN loss (can occur early in 4-bit training)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  [sft] WARNING: NaN/Inf loss at step {step}, skipping batch")
                    optimizer.zero_grad()
                    continue

                # Add phonetic head loss if available
                if "phoneme_logits" in out and "phoneme_ids" in batch:
                    ph_loss = phonetic_head_loss(out["phoneme_logits"], batch["phoneme_ids"])
                    if not (torch.isnan(ph_loss) or torch.isinf(ph_loss)):
                        loss = loss + phonetic_loss_weight * ph_loss

                # Scale loss for gradient accumulation
                scaled_loss = loss / grad_accum_steps
                accelerator.backward(scaled_loss)

                if not is_accumulation_step:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1
                total_loss += loss.item()

                if accelerator.is_main_process:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

                # Step-based checkpoint saving
                if save_steps > 0 and global_step % save_steps == 0 and accelerator.is_main_process:
                    step_ckpt_path = f"{output_subdir}/step_{global_step}"
                    save_checkpoint(model, tokenizer, step_ckpt_path)

            # Validation
            if val_dl:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_dl:
                        out = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                        val_loss += out["lm_loss"].item()
                val_loss /= len(val_dl)
                if accelerator.is_main_process:
                    print(f"  Val loss: {val_loss:.4f}")

            # Save checkpoint
            if accelerator.is_main_process:
                ckpt_path = f"{output_subdir}/epoch_{epoch+1}"
                save_checkpoint(model, tokenizer, ckpt_path)
                last_saved_epoch = epoch + 1

    except KeyboardInterrupt:
        interrupted = True
        print("\n[sft] Interrupted — saving emergency checkpoint...")
        if accelerator.is_main_process:
            # Save at the current step within the interrupted epoch
            partial_epoch_label = f"epoch_{last_saved_epoch + 1}_partial_step_{global_step}"
            ckpt_path = f"{output_subdir}/{partial_epoch_label}"
            try:
                save_checkpoint(model, tokenizer, ckpt_path)
                print(f"[sft] Trained {global_step} steps total before interrupt")
            except Exception as save_exc:
                print(f"[sft] WARNING: emergency save failed: {save_exc}")

    if accelerator.is_main_process:
        if interrupted:
            last_ckpt = (
                f"{output_subdir}/epoch_{last_saved_epoch + 1}_partial_step_{global_step}"
                if last_saved_epoch < epochs
                else f"{output_subdir}/epoch_{last_saved_epoch}"
            )
            print(f"\nInterrupted at step {global_step}. Last checkpoint: {last_ckpt}")
        else:
            print(f"\nTraining complete. Final checkpoint: {output_subdir}/epoch_{epochs}")


if __name__ == "__main__":
    import typer
    app = typer.Typer()

    @app.command()
    def main(
        stage: int = typer.Option(1),
        genre: Optional[str] = typer.Option(None),
        data: str = typer.Option("data/raw/all_songs.jsonl"),
        val: Optional[str] = typer.Option(None),
        base_model: str = typer.Option("mistralai/Mistral-7B-Instruct-v0.2"),
        output_dir: str = typer.Option("checkpoints"),
        batch_size: int = typer.Option(2),
        epochs: int = typer.Option(1),
        lr: float = typer.Option(1e-4),
        lora_rank: int = typer.Option(32),
        no_4bit: bool = typer.Option(False),
        training_mode: str = typer.Option("ccl", help="'ccl' or 'simple'"),
        save_steps: int = typer.Option(200, help="Save checkpoint every N steps"),
        max_per_artist: int = typer.Option(3, help="Cap songs per artist"),
    ):
        train_sft(
            stage=stage, genre=genre, data_path=data, val_path=val,
            base_model=base_model, output_dir=output_dir,
            batch_size=batch_size, epochs=epochs, lr=lr,
            lora_rank=lora_rank, use_4bit=not no_4bit,
            training_mode=training_mode, save_steps=save_steps,
            max_per_artist=max_per_artist,
        )

    app()
