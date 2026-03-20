"""
Felt State Engine
=================
Models the pre-verbal, embodied emotional state that drives lyric generation
in human writers.

THE NEUROSCIENCE
----------------
When a rapper sits down to write, the beat enters their body before a single
word exists. This creates a *felt state* — a somatic, pre-linguistic emotional
configuration (Damasio, 1994; "Descartes' Error").  Words don't create the
emotion; the emotion creates the words.

Current systems get this backwards: they generate text first, then score the
emotion.  The felt state inverts that — emotion is the *input*, not the output.

TWO MECHANISMS
--------------
1. Emotional contagion: as each line is accepted, the writer gets infected by
   their own output.  Writing dark lines makes you feel darker.  Writing a
   victorious hook shifts your internal state toward triumph.  The state feeds
   forward into the next line's generation context.

2. Genre gravitational pull: the target emotional trajectory for the genre/
   section acts as a gravitational attractor.  The felt state drifts toward
   the target but never snaps instantly — it has inertia, like a real body.

PROMPT INJECTION
----------------
The felt state is converted to emotion tokens injected directly into the
generation prompt, so the LLM conditions on the current emotional body state
from token 1 — not as a post-hoc scoring criterion.

Example prompt fragment injected before lyrics context:
  [FELT: tension=0.72 valence=-0.31 arousal=0.68 intimacy=0.45]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.model.emotional_geometry import EmotionPoint, line_emotion, get_target_point


# ── Felt state dataclass ──────────────────────────────────────────────────────

@dataclass
class FeltState:
    """
    The writer's current embodied emotional state.
    Updated after each accepted line via emotional contagion.
    Gravitates toward genre trajectory target.
    """
    valence:   float = 0.0   # [-1, 1]  negative ↔ positive
    arousal:   float = 0.3   # [-1, 1]  calm ↔ energised
    tension:   float = 0.2   # [0, 1]   resolved ↔ tense
    intimacy:  float = 0.3   # [-1, 1]  distant ↔ confessional
    momentum:  float = 0.0   # [-1, 1]  falling ↔ rising (trajectory direction)
    intensity: float = 0.5   # [0, 1]   how strongly the state is felt

    # Emotional contagion history: last N accepted lines' valences
    _valence_history: list[float] = field(default_factory=list, repr=False)

    def to_prompt_token(self) -> str:
        """
        Render as a compact token for prompt injection.
        The model learns to condition on this during training.
        """
        t_str = f"{self.tension:.2f}"
        v_str = f"{self.valence:+.2f}"
        a_str = f"{self.arousal:.2f}"
        i_str = f"{self.intimacy:.2f}"
        m_str = "rising" if self.momentum > 0.1 else ("falling" if self.momentum < -0.1 else "flat")
        return f"[FELT: tension={t_str} valence={v_str} arousal={a_str} intimacy={i_str} momentum={m_str}]"

    def to_array(self) -> np.ndarray:
        return np.array([self.valence, self.arousal, self.tension, self.intimacy,
                         self.momentum, self.intensity], dtype=np.float32)

    @classmethod
    def from_genre_section(cls, genre: str, section: str) -> "FeltState":
        """Initialise felt state from genre trajectory target for a section."""
        target = get_target_point(genre, section)
        return cls(
            valence=float(target.valence),
            arousal=float(target.arousal),
            tension=float(target.tension),
            intimacy=float(target.intimacy),
            momentum=0.0,
            intensity=0.5,
        )


# ── Felt state engine ─────────────────────────────────────────────────────────

class FeltStateEngine:
    """
    Manages the writer's embodied emotional state across a song.

    Two update mechanisms:
      1. contagion_update(line) — called after each accepted line
      2. section_reset(genre, section) — resets toward section target at transitions
    """

    # How strongly each accepted line infects the felt state (0=no effect, 1=instant)
    CONTAGION_RATE    = 0.28

    # How strongly the genre/section target pulls the felt state (gravity)
    GRAVITY_RATE      = 0.12

    # Inertia: felt state resists sudden changes (like a real body)
    INERTIA           = 0.60

    def __init__(self, genre: str, section: str = "intro"):
        self.state = FeltState.from_genre_section(genre, section)
        self.genre = genre
        self.current_section = section
        self._line_count = 0
        self._prev_valence = self.state.valence

    def contagion_update(self, accepted_line: str) -> FeltState:
        """
        Update felt state via emotional contagion from an accepted line.
        The writer's body state shifts toward the emotional content of
        their own output — writing darkness makes you feel darker.
        """
        ep: EmotionPoint = line_emotion(accepted_line)
        self._line_count += 1

        # Contagion: blend line emotion into current state
        alpha = self.CONTAGION_RATE
        new_valence  = self.state.valence  * (1 - alpha) + float(ep.valence)  * alpha
        new_arousal  = self.state.arousal  * (1 - alpha) + float(ep.arousal)  * alpha
        new_tension  = self.state.tension  * (1 - alpha) + float(ep.tension)  * alpha
        new_intimacy = self.state.intimacy * (1 - alpha) + float(ep.intimacy) * alpha

        # Gravity: pull toward genre/section target
        target = get_target_point(self.genre, self.current_section)
        g = self.GRAVITY_RATE
        new_valence  = new_valence  * (1 - g) + float(target.valence)  * g
        new_arousal  = new_arousal  * (1 - g) + float(target.arousal)  * g
        new_tension  = new_tension  * (1 - g) + float(target.tension)  * g
        new_intimacy = new_intimacy * (1 - g) + float(target.intimacy) * g

        # Momentum: direction of valence travel
        momentum = float(np.clip(new_valence - self._prev_valence, -1.0, 1.0))

        # Intensity: how strongly the state is currently felt
        # Peaks when tension is high and there's strong momentum
        intensity = float(np.clip(
            0.4 + abs(momentum) * 0.3 + self.state.tension * 0.3,
            0.0, 1.0
        ))

        self.state._valence_history.append(new_valence)
        if len(self.state._valence_history) > 8:
            self.state._valence_history.pop(0)

        self._prev_valence = new_valence
        self.state = FeltState(
            valence=float(np.clip(new_valence,  -1, 1)),
            arousal=float(np.clip(new_arousal,  -1, 1)),
            tension=float(np.clip(new_tension,   0, 1)),
            intimacy=float(np.clip(new_intimacy, -1, 1)),
            momentum=float(np.clip(momentum,     -1, 1)),
            intensity=intensity,
            _valence_history=list(self.state._valence_history),
        )
        return self.state

    def section_transition(self, new_section: str) -> FeltState:
        """
        At section boundaries, pull felt state strongly toward new section target.
        Mirrors the cognitive shift a writer makes when moving from verse to chorus.
        """
        self.current_section = new_section
        target = get_target_point(self.genre, new_section)

        # Section transitions are more abrupt than line-by-line contagion
        alpha = 0.45
        self.state = FeltState(
            valence=float(np.clip(
                self.state.valence * (1 - alpha) + float(target.valence) * alpha, -1, 1)),
            arousal=float(np.clip(
                self.state.arousal * (1 - alpha) + float(target.arousal) * alpha, -1, 1)),
            tension=float(np.clip(
                self.state.tension * (1 - alpha) + float(target.tension) * alpha,  0, 1)),
            intimacy=float(np.clip(
                self.state.intimacy * (1 - alpha) + float(target.intimacy) * alpha, -1, 1)),
            momentum=0.0,
            intensity=0.6,
            _valence_history=list(self.state._valence_history),
        )
        self._line_count = 0
        return self.state

    def emotional_temperature(self) -> float:
        """
        Derive a generation temperature modifier from the felt state.
        High intensity + high tension → hotter generation (more uninhibited)
        Calm, resolved state → cooler (more deliberate)
        Mirrors how emotional arousal affects verbal fluency in humans.
        """
        base = 0.85
        intensity_boost = self.state.intensity * 0.20
        tension_boost   = self.state.tension   * 0.15
        return float(np.clip(base + intensity_boost + tension_boost, 0.6, 1.3))

    def describe(self) -> str:
        """Human-readable state description for logging."""
        v = "positive" if self.state.valence > 0.2 else ("negative" if self.state.valence < -0.2 else "neutral")
        a = "high energy" if self.state.arousal > 0.5 else "low energy"
        t = f"tension={self.state.tension:.2f}"
        m = "↑" if self.state.momentum > 0.1 else ("↓" if self.state.momentum < -0.1 else "→")
        return f"[FeltState] {v}, {a}, {t}, momentum={m}, intensity={self.state.intensity:.2f}"
