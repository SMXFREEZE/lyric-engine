"""Genre definitions, special tokens, and their LoRA adapter configs.

GENRES lists every style the system recognises.  A genre can be used for:
  1. LoRA adapter training (when a dataset exists)
  2. Style DNA conditioning at inference
  3. API routing

The GENRES list is synchronised with src/data/style_dna.py so every style
with a StyleDNA entry is a valid generation target.
"""

# ── Supported genres (synchronised with style_dna.py) ─────────────────────────
# Every entry here must have a matching GENRE_DESCRIPTIONS entry.

GENRES = [
    # Hip-Hop / Rap
    "trap",
    "drill",
    "hip_hop",
    "boom_bap",
    "uk_rap",
    "grime",
    "latin_trap",
    "french_rap",
    # R&B / Soul
    "rnb",
    "neo_soul",
    # Pop
    "pop",
    "indie_pop",
    # Afrobeats / Afropop
    "afrobeats",
    "afroswing",
    "amapiano",
    # Latin
    "reggaeton",
    "bachata",
    "corrido",
    # K-Pop / Asian
    "kpop",
    "j_pop",
    # Arabic / Middle East
    "arabic_pop",
    "mahraganat",
    # Moroccan / North African
    "chaabi",
    "rai",
    # Dancehall / Reggae
    "dancehall",
    # Indie / Alternative
    "indie",
    "alt_emo",
    # Electronic / EDM
    "edm",
    # Country / Rock
    "country",
    "rock",
]

GENRE_DESCRIPTIONS = {
    # Hip-Hop / Rap
    "trap":       "808 cadence, triplet flow, flex imagery, dark production",
    "drill":      "UK/Chicago variants, street-specific slang, threat posturing, slide references",
    "hip_hop":    "Wordplay, cultural references, storytelling, boom-bap cadence",
    "boom_bap":   "Golden-era flow, multisyllabic rhymes, sample-heavy beats, lyrical depth",
    "uk_rap":     "Fast-paced delivery, London slang, roadman culture, sliding cadence",
    "grime":      "140 BPM, rapid staccato delivery, UK energy, clash-style bars",
    "latin_trap": "Spanish/Spanglish, triplet and reggaeton flows, Latin street culture",
    "french_rap": "French language, multisyllabic rhyme depth, banlieue storytelling",
    # R&B / Soul
    "rnb":        "Melisma-friendly lines, vulnerability, second-person intimacy, gospel callbacks",
    "neo_soul":   "Free-flowing melodies, jazz harmonies, conscious lyrics, organic instrumentation",
    # Pop
    "pop":        "Hook-first structure, universal themes, short syllabic lines, ear-worm phrasing",
    "indie_pop":  "Concrete imagery, non-obvious metaphors, conversational melodic delivery",
    # Afrobeats / Afropop
    "afrobeats":  "Syncopated groove, call-response, Pidgin/Yoruba code-switching, dance energy",
    "afroswing":  "UK-African fusion, melodic rap, Afro-influenced swing, multicultural vibes",
    "amapiano":   "Log drum bass, chanted vocals, Zulu/Sesotho lyrics, house-influenced",
    # Latin
    "reggaeton":  "Dembow riddim, Spanish lyrics, club energy, rhythmic wordplay",
    "bachata":    "Romantic mellowness, Spanish lyrics, acoustic guitar, heartbreak themes",
    "corrido":    "Corridos tumbados, narrative storytelling, Spanish, norteño-trap fusion",
    # K-Pop / Asian
    "kpop":       "Korean/English blend, precise pop structure, rap breaks, performance energy",
    "j_pop":      "Japanese lyrics, anime-influenced melodics, high BPM, catchy hooks",
    # Arabic / Middle East
    "arabic_pop": "Arabic lyrics, ornamental melodies, Middle Eastern scales, romantic themes",
    "mahraganat": "Egyptian street rap, rapid Arabic delivery, chanted hooks, high energy",
    # Moroccan / North African
    "chaabi":     "Moroccan folk-pop, Arabic/Berber/French mix, call-response, festive energy",
    "rai":        "Raï modernised, Arabic/French, improvisational melodies, North African groove",
    # Dancehall / Reggae
    "dancehall":  "Jamaican patois, syncopated riddims, dancehall energy, Caribbean vibes",
    # Indie / Alternative
    "indie":      "Concrete imagery, lowercase aesthetic, non-obvious metaphors, emotional restraint",
    "alt_emo":    "Self-referential, stream of consciousness, distorted metaphors, introspective arcs",
    # Electronic / EDM
    "edm":        "Drop structure, minimal lyrics, euphoric energy, festival anthems",
    # Country / Rock
    "country":    "Narrative storytelling, rural imagery, heartbreak/pride themes, twang phrasing",
    "rock":       "Power chords energy, rebellion, anthemic hooks, distortion imagery",
}

# ── Emotional arc control tokens ──────────────────────────────────────────────
ARC_TOKENS = [
    "[SETUP]", "[BUILD]", "[RELEASE]", "[REFRAME]", "[PEAK]", "[OUTRO]",
]

# ── Song structure tokens ─────────────────────────────────────────────────────
# Includes non-English section names for Latin, French, and Afrobeats structures
SECTION_TOKENS = [
    # English
    "[VERSE]", "[PRECHORUS]", "[CHORUS]", "[BRIDGE]", "[HOOK]", "[OUTRO]", "[INTRO]",
    # Latin / Spanish
    "[VERSO]", "[CORO]", "[PUENTE]",
    # French
    "[COUPLET]", "[REFRAIN]", "[PONTE]",
]

# ── Flow-specific tokens ─────────────────────────────────────────────────────
FLOW_TOKENS = [
    "[TRIPLET_FLOW]", "[DEMBOW]", "[CALL_RESPONSE]", "[DOUBLE_TIME]",
    "[SYNCOPATED]", "[MELODIC]", "[STACCATO]", "[NARRATIVE]",
]

# All special tokens combined
SPECIAL_TOKENS = ARC_TOKENS + SECTION_TOKENS + FLOW_TOKENS + [
    "[GENRE_START]", "[GENRE_END]",
    "[STYLE_START]", "[STYLE_END]",
    "[RHYME_AABB]", "[RHYME_ABAB]", "[RHYME_ABCB]", "[RHYME_FREE]",
]

# ── LoRA configuration ────────────────────────────────────────────────────────

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

BASE_MODEL_LORA_CONFIG = {
    **LORA_CONFIG,
    "r": 64,  # Wider for the general music SFT adapter
    "lora_alpha": 128,
}
