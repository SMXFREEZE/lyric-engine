"""
Multi-provider lyrics retrieval.

Provider order is quality-aware rather than single-source:
  1. Genius direct lookup (if token + song id available)
  2. Vagalume API
  3. LRCLIB
  4. lyrics.ovh
  5. Genius search fallback

The caller can fetch all candidates and keep the highest-quality version.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import requests


GENIUS_TOKEN = os.getenv("GENIUS_TOKEN", "")
VAGALUME_API_KEY = os.getenv("VAGALUME_API_KEY", "")

REQUEST_HEADERS = {
    "User-Agent": "LyricEngine/1.0 (+https://github.com/SMXFREEZE/lyric-engine)",
    "Accept": "application/json,text/plain,*/*",
}

PROVIDER_WEIGHTS = {
    "genius_id": 1.00,
    "vagalume": 0.95,
    "lrclib": 0.88,
    "lyrics_ovh": 0.78,
    "genius_search": 0.74,
}


@dataclass
class LyricsCandidate:
    artist: str
    title: str
    lyrics: str
    source: str
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record_fields(self) -> dict[str, Any]:
        return {
            "lyrics_source": self.source,
            "lyrics_quality": round(self.quality_score, 4),
            "lyrics_meta": self.metadata,
        }


def _normalize_space(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def _strip_timestamps(text: str) -> str:
    return re.sub(r"(?m)^\s*(?:\[\d{1,2}:\d{2}(?:\.\d{1,2})?\]|\d{1,2}:\d{2}(?:\.\d{1,2})?)\s*", "", text)


def clean_lyrics_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = _strip_timestamps(text)
    text = re.sub(r"\[?Instrumental\]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[?Intro\]?", "Intro", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [_normalize_space(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _estimate_quality(lyrics: str, source: str) -> float:
    words = _tokenize(lyrics)
    if not words:
        return 0.0

    unique_ratio = len(set(words)) / max(len(words), 1)
    lines = [line for line in lyrics.splitlines() if line.strip()]
    avg_line_words = len(words) / max(len(lines), 1)
    repetition_penalty = max(0.0, 0.16 - unique_ratio)

    score = PROVIDER_WEIGHTS.get(source, 0.60)
    score += min(len(words), 700) / 1600.0
    score += min(len(lines), 80) / 400.0
    score += min(unique_ratio, 0.82) / 5.0
    if 2 <= avg_line_words <= 14:
        score += 0.04
    score -= repetition_penalty
    return round(max(score, 0.0), 4)


def rank_lyrics_candidates(candidates: list[LyricsCandidate]) -> list[LyricsCandidate]:
    return sorted(candidates, key=lambda item: item.quality_score, reverse=True)


def _make_candidate(
    artist: str,
    title: str,
    lyrics: str,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> LyricsCandidate | None:
    cleaned = clean_lyrics_text(lyrics)
    if len(_tokenize(cleaned)) < 40:
        return None
    return LyricsCandidate(
        artist=artist,
        title=title,
        lyrics=cleaned,
        source=source,
        quality_score=_estimate_quality(cleaned, source),
        metadata=metadata or {},
    )


def _get_json(url: str, *, params: dict[str, Any] | None = None, timeout: float = 12.0) -> Any | None:
    try:
        response = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=timeout)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:
        return None


def _fetch_genius_by_song_id(artist: str, title: str, genius_song_id: int) -> LyricsCandidate | None:
    if not GENIUS_TOKEN or not genius_song_id:
        return None

    try:
        import lyricsgenius

        genius = lyricsgenius.Genius(
            GENIUS_TOKEN,
            verbose=False,
            remove_section_headers=True,
            skip_non_songs=True,
            timeout=10,
        )
        song = genius.song(genius_song_id)
    except Exception:
        return None

    if not song or not getattr(song, "lyrics", None):
        return None

    return _make_candidate(
        artist,
        title,
        song.lyrics,
        "genius_id",
        {"genius_song_id": genius_song_id},
    )


def _fetch_genius_search(artist: str, title: str) -> LyricsCandidate | None:
    if not GENIUS_TOKEN:
        return None

    try:
        import lyricsgenius

        genius = lyricsgenius.Genius(
            GENIUS_TOKEN,
            verbose=False,
            remove_section_headers=True,
            skip_non_songs=True,
            timeout=10,
        )
        song = genius.search_song(title, artist)
    except Exception:
        return None

    if not song or not getattr(song, "lyrics", None):
        return None

    return _make_candidate(
        artist,
        title,
        song.lyrics,
        "genius_search",
        {"genius_title": getattr(song, "title", title)},
    )


def _fetch_vagalume(artist: str, title: str) -> LyricsCandidate | None:
    if not VAGALUME_API_KEY:
        return None

    data = _get_json(
        "https://api.vagalume.com.br/search.php",
        params={
            "art": artist,
            "mus": title,
            "apikey": VAGALUME_API_KEY,
        },
    )
    if not isinstance(data, dict):
        return None

    songs = data.get("mus") or []
    if not songs:
        return None

    best_song = next((item for item in songs if item.get("text")), songs[0])
    lyrics = best_song.get("text")
    if not lyrics:
        return None

    return _make_candidate(
        artist,
        title,
        lyrics,
        "vagalume",
        {
            "match_type": data.get("type"),
            "song_id": best_song.get("id"),
            "url": best_song.get("url"),
        },
    )


def _lrclib_lyrics_from_payload(payload: dict[str, Any]) -> str | None:
    plain = payload.get("plainLyrics")
    if plain:
        return plain
    synced = payload.get("syncedLyrics")
    if synced:
        return _strip_timestamps(synced)
    return None


def _fetch_lrclib(artist: str, title: str) -> LyricsCandidate | None:
    data = _get_json(
        "https://lrclib.net/api/get",
        params={
            "artist_name": artist,
            "track_name": title,
        },
    )
    payloads: list[dict[str, Any]] = []

    if isinstance(data, dict):
        payloads.append(data)
    else:
        search_data = _get_json(
            "https://lrclib.net/api/search",
            params={
                "artist_name": artist,
                "track_name": title,
            },
        )
        if isinstance(search_data, list):
            payloads.extend(item for item in search_data if isinstance(item, dict))

    for payload in payloads:
        lyrics = _lrclib_lyrics_from_payload(payload)
        if not lyrics:
            continue
        candidate = _make_candidate(
            artist,
            title,
            lyrics,
            "lrclib",
            {
                "lrclib_id": payload.get("id"),
                "instrumental": payload.get("instrumental"),
            },
        )
        if candidate:
            return candidate
    return None


def _fetch_lyrics_ovh(artist: str, title: str) -> LyricsCandidate | None:
    url = f"https://api.lyrics.ovh/v1/{quote(artist)}/{quote(title)}"
    data = _get_json(url)
    if not isinstance(data, dict):
        return None
    lyrics = data.get("lyrics")
    if not lyrics:
        return None
    return _make_candidate(artist, title, lyrics, "lyrics_ovh")


def available_lyrics_sources() -> list[str]:
    sources: list[str] = []
    if GENIUS_TOKEN:
        sources.extend(["genius_id", "genius_search"])
    if VAGALUME_API_KEY:
        sources.append("vagalume")
    sources.extend(["lrclib", "lyrics_ovh"])
    return sources


def fetch_lyrics_candidates(
    artist: str,
    title: str,
    *,
    genius_song_id: int | None = None,
) -> list[LyricsCandidate]:
    candidates: list[LyricsCandidate] = []

    if genius_song_id:
        candidate = _fetch_genius_by_song_id(artist, title, genius_song_id)
        if candidate:
            candidates.append(candidate)

    for fetcher in (_fetch_vagalume, _fetch_lrclib, _fetch_lyrics_ovh):
        candidate = fetcher(artist, title)
        if candidate:
            candidates.append(candidate)

    if not candidates:
        candidate = _fetch_genius_search(artist, title)
        if candidate:
            candidates.append(candidate)

    return rank_lyrics_candidates(candidates)


def fetch_best_lyrics(
    artist: str,
    title: str,
    *,
    genius_song_id: int | None = None,
) -> LyricsCandidate | None:
    candidates = fetch_lyrics_candidates(artist, title, genius_song_id=genius_song_id)
    return candidates[0] if candidates else None
