"""
Multi-provider lyrics scraper.

Discovery:
  1. Deezer artist top tracks (primary, no auth)
  2. Genius artist songs (fallback, if token works)

Lyrics retrieval:
  1. Genius direct song lookup
  2. Vagalume
  3. LRCLIB
  4. lyrics.ovh
  5. Genius search fallback

Output: JSONL files with {artist, title, genre, lyrics} per song.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

import lyricsgenius
import requests
from tqdm import tqdm

from src.data.lyrics_providers import fetch_best_lyrics

GENIUS_TOKEN = os.getenv("GENIUS_TOKEN", "")
DEEZER_BASE = "https://api.deezer.com"
DEEZER_HEADERS = {"User-Agent": "LyricEngine/1.0"}

# Curated seed artists per genre — add more as needed
GENRE_SEEDS: dict[str, list[str]] = {
    "trap":    ["Future", "Young Thug", "Gunna", "Lil Baby", "21 Savage", "Roddy Ricch"],
    "rnb":     ["Frank Ocean", "SZA", "H.E.R.", "Bryson Tiller", "Summer Walker", "The Weeknd"],
    "indie":   ["Phoebe Bridgers", "boygenius", "Sufjan Stevens", "Bon Iver", "Mitski", "Clairo"],
    "pop":     ["Taylor Swift", "Olivia Rodrigo", "Dua Lipa", "Ariana Grande", "Harry Styles"],
    "drill":   ["Pop Smoke", "Fivio Foreign", "Lil Durk", "King Von", "Central Cee", "Digga D"],
    "alt_emo": ["Paramore", "My Chemical Romance", "Lana Del Rey", "Billie Eilish", "Halsey"],
    "hip_hop": ["Kendrick Lamar", "J. Cole", "Drake", "Jay-Z", "Nas", "Lupe Fiasco"],
    "country": ["Morgan Wallen", "Luke Combs", "Zach Bryan", "Kacey Musgraves", "Tyler Childers"],
    "rock":    ["Arctic Monkeys", "The Strokes", "Tame Impala", "Radiohead", "Foo Fighters"],
    "latin":   ["Bad Bunny", "J Balvin", "Ozuna", "Karol G", "Peso Pluma"],
}

def deezer_get(path: str, params: Optional[dict] = None) -> dict | None:
    try:
        response = requests.get(
            f"{DEEZER_BASE}{path}",
            params=params or {},
            headers=DEEZER_HEADERS,
            timeout=10,
        )
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:
        return None


def deezer_top_tracks(artist_name: str, limit: int = 50) -> list[dict]:
    search = deezer_get("/search/artist", {"q": artist_name})
    if not search:
        return []

    data = search.get("data") or []
    if not data:
        return []

    artist_id = data[0].get("id")
    if not artist_id:
        return []

    top = deezer_get(f"/artist/{artist_id}/top", {"limit": min(limit, 100)})
    tracks = top.get("data") if top else []
    results = []
    for track in tracks or []:
        title = track.get("title")
        if not title:
            continue
        results.append(
            {
                "title": title,
                "deezer_track_id": track.get("id"),
            }
        )
    return results


def genius_top_tracks(
    genius: Optional[lyricsgenius.Genius],
    artist_name: str,
    max_songs: int,
) -> list[dict]:
    if not genius:
        return []
    try:
        artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
        if not artist:
            return []
        return [
            {
                "title": song.title,
                "genius_song_id": getattr(song, "id", None),
            }
            for song in artist.songs
            if getattr(song, "title", None)
        ]
    except Exception:
        return []


def discover_tracks(
    genius: Optional[lyricsgenius.Genius],
    artist_name: str,
    max_songs: int,
) -> list[dict]:
    seen_titles: set[str] = set()
    tracks: list[dict] = []

    for provider_tracks in (
        deezer_top_tracks(artist_name, max_songs),
        genius_top_tracks(genius, artist_name, max_songs),
    ):
        for item in provider_tracks:
            title = item.get("title", "").strip()
            key = title.lower()
            if not title or key in seen_titles:
                continue
            seen_titles.add(key)
            tracks.append(item)
            if len(tracks) >= max_songs:
                return tracks
    return tracks


def scrape_artist(
    genius: Optional[lyricsgenius.Genius],
    artist_name: str,
    genre: str,
    max_songs: int = 50,
    out_dir: Path = Path("data/raw"),
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        tracks = discover_tracks(genius, artist_name, max_songs)
        if not tracks:
            return results

        for song in tracks:
            candidate = fetch_best_lyrics(
                artist_name,
                song["title"],
                genius_song_id=song.get("genius_song_id"),
            )
            if not candidate or len(candidate.lyrics.split()) < 80:
                continue
            record = {
                "artist": artist_name,
                "title": song["title"],
                "genre": genre,
                "lyrics": candidate.lyrics,
            }
            record.update(candidate.to_record_fields())
            results.append(record)
    except Exception as e:
        print(f"Error scraping {artist_name}: {e}")
    return results


def run_scrape(
    genres: Optional[list[str]] = None,
    max_songs_per_artist: int = 50,
    out_dir: str = "data/raw",
    token: str = GENIUS_TOKEN,
):
    genius = None
    if token:
        genius = lyricsgenius.Genius(token, verbose=False, remove_section_headers=True, skip_non_songs=True)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_genres = genres or list(GENRE_SEEDS.keys())
    all_records: list[dict] = []

    for genre in target_genres:
        artists = GENRE_SEEDS.get(genre, [])
        print(f"\n=== Genre: {genre} ({len(artists)} artists) ===")
        genre_records = []
        for artist_name in tqdm(artists, desc=genre):
            records = scrape_artist(genius, artist_name, genre, max_songs_per_artist, out_path)
            genre_records.extend(records)
            time.sleep(1)  # rate limit

        # Write per-genre JSONL
        genre_file = out_path / f"{genre}.jsonl"
        with open(genre_file, "w", encoding="utf-8") as f:
            for r in genre_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(genre_records)} songs → {genre_file}")
        all_records.extend(genre_records)

    # Write combined
    combined_file = out_path / "all_songs.jsonl"
    with open(combined_file, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nTotal: {len(all_records)} songs → {combined_file}")
    return all_records


if __name__ == "__main__":
    import typer
    typer.run(run_scrape)
