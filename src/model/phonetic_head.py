"""
Phonetic constraint head.

A lightweight 2-layer MLP trained to predict phoneme IDs from hidden states.

At inference time it can rescore beam candidates and penalize continuations that
drift away from the target rhyme phoneme or stress pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.dual_tokenizer import PHONEME_TO_ID, PHONEME_VOCAB_SIZE


class PhoneticHead(nn.Module):
    def __init__(self, d_model: int = 4096, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, PHONEME_VOCAB_SIZE),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (batch, seq_len, d_model)
        returns logits: (batch, seq_len, PHONEME_VOCAB_SIZE)
        """
        return self.net(hidden_states)

    def predict_last_phoneme(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (batch, seq_len, d_model)
        returns predicted phoneme IDs for the last token: (batch,)
        """
        last = hidden_states[:, -1, :]
        logits = self.net(last)
        return logits.argmax(dim=-1)


class PhoneticConstraintScorer:
    """
    Wraps PhoneticHead and provides beam candidate reranking.
    """

    def __init__(self, head: PhoneticHead, device: str = "cpu"):
        self.head = head.to(device)
        self.head.eval()
        self.device = device

    @torch.no_grad()
    def score_candidates(
        self,
        hidden_states: torch.Tensor,
        target_end_phoneme_id: int,
        stress_targets: list[int] | None = None,
        rhyme_weight: float = 2.0,
        stress_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Returns a penalty tensor of shape (beam,). Higher penalty is worse.
        """
        logits = self.head(hidden_states)

        last_logits = logits[:, -1, :]
        last_probs = F.softmax(last_logits, dim=-1)
        rhyme_prob = last_probs[:, target_end_phoneme_id]
        rhyme_penalty = -rhyme_weight * torch.log(rhyme_prob.clamp(min=1e-9))

        penalty = rhyme_penalty

        if stress_targets is not None:
            stressed_ids = [idx for token, idx in PHONEME_TO_ID.items() if token.endswith("1")]
            if stressed_ids:
                for pos, target_stress in enumerate(stress_targets):
                    if pos >= logits.shape[1]:
                        break
                    if target_stress != 1:
                        continue
                    pos_probs = F.softmax(logits[:, pos, :], dim=-1)
                    stress_prob = pos_probs[:, stressed_ids].sum(dim=-1)
                    penalty += stress_weight * (-torch.log(stress_prob.clamp(min=1e-9)))

        return penalty

    def rerank_beams(
        self,
        beam_hidden_states: torch.Tensor,
        beam_scores: torch.Tensor,
        target_end_phoneme_id: int,
        alpha: float = 0.5,
    ) -> list[int]:
        """
        Returns sorted beam indices (best first).
        """
        penalty = self.score_candidates(beam_hidden_states, target_end_phoneme_id)
        lm_norm = (beam_scores - beam_scores.mean()) / (beam_scores.std() + 1e-9)
        ph_norm = (penalty - penalty.mean()) / (penalty.std() + 1e-9)
        combined = alpha * lm_norm - (1 - alpha) * ph_norm
        return combined.argsort(descending=True).tolist()


def phonetic_head_loss(
    logits: torch.Tensor,
    phoneme_ids: torch.Tensor,
    ignore_index: int = 0,
) -> torch.Tensor:
    """
    Cross-entropy loss for phonetic head training.

    When style conditioning is enabled, the model prepends one learned prefix
    token to the sequence. In that case the phoneme target stream is shorter by
    one position, so we left-pad it with ignore_index to keep the alignment.
    """
    batch, seq, vocab = logits.shape
    if phoneme_ids.ndim != 2:
        raise ValueError(f"Expected phoneme_ids rank 2, got shape {tuple(phoneme_ids.shape)}")

    target = phoneme_ids.to(logits.device)
    target_seq = target.shape[1]

    if target_seq < seq:
        pad = torch.full(
            (target.shape[0], seq - target_seq),
            ignore_index,
            dtype=target.dtype,
            device=target.device,
        )
        target = torch.cat([pad, target], dim=1)
    elif target_seq > seq:
        target = target[:, :seq]

    return F.cross_entropy(
        logits.reshape(batch * seq, vocab),
        target.reshape(batch * seq),
        ignore_index=ignore_index,
    )


if __name__ == "__main__":
    head = PhoneticHead(d_model=256, hidden=128)
    dummy_hidden = torch.randn(4, 10, 256)
    logits = head(dummy_hidden)
    print(f"Logits shape: {logits.shape}")

    scorer = PhoneticConstraintScorer(head)
    beam_scores = torch.tensor([-1.2, -1.5, -0.8, -2.1])
    target_id = PHONEME_TO_ID.get("EY1", 10)
    ranked = scorer.rerank_beams(dummy_hidden, beam_scores, target_id)
    print(f"Reranked beam order: {ranked}")
