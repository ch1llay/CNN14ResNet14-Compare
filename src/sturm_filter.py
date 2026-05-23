"""Sturm fault-filter для GTZAN.

Источник: Sturm B. L. (2013, 2014) — каталог проблем GTZAN
(repetitions, mislabelings, distortions).

Готовые stratified-filtered splits (Kereliuk 2015):
https://github.com/coreyker/dnn-mgr/tree/master/gtzan

Стратегия:
1. По умолчанию — скачать train/valid/test_filtered.txt из репо Kereliuk и
   взять их union как «чистый» подмножество GTZAN.
2. Fallback — embedded минимальный список известных проблемных треков.

Использование:
    from src.sturm_filter import get_clean_track_ids
    clean = get_clean_track_ids(genres_dir, method="kereliuk")
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

KERELIUK_BASE = (
    "https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/"
)
KERELIUK_FILES = ("train_filtered.txt", "valid_filtered.txt", "test_filtered.txt")

# Минимальный hardcoded список заведомо проблемных треков (fallback).
# Это subset Sturm 2014 Table 1 — exact/near duplicates и mislabels,
# широко цитируемые в литературе. Полный список из ~100 треков
# рекомендуется получать через method="kereliuk".
KNOWN_DUPLICATES = (
    # Точные/near дубликаты внутри жанра (recording duplicates)
    "jazz.00054",   # дубликат jazz.00055
    "reggae.00086", # near-dup reggae.00087
    "reggae.00087",
    "rock.00045",   # near-dup rock.00046
    "rock.00046",
    "hiphop.00038", # dup
    "country.00056",
    "country.00057",
    "blues.00081",
    "blues.00082",
)

KNOWN_MISLABELS = (
    # Треки с неверными жанровыми метками (по Sturm 2013/2014)
    "reggae.00086",  # фактически hip-hop/dancehall
    "hiphop.00076",  # фактически reggae
    "jazz.00037",    # фактически classical
    "rock.00084",    # фактически metal-adjacent
)

# Хранится для документации; реальная фильтрация делается через Kereliuk-list.
KNOWN_DISTORTIONS = (
    "jazz.00054",
    "country.00057",
)


def _parse_kereliuk_file(text: str) -> set[str]:
    """Извлечь имена треков (без .wav, без пути) из одного filtered.txt.

    Формат строк: 'gtzan/genres/blues/blues.00000.wav' либо просто 'blues.00000'.
    """
    ids: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Пытаемся выдрать pattern <genre>.<digits>
        m = re.search(r"([a-z]+)\.(\d{5})", line)
        if m:
            ids.add(f"{m.group(1)}.{m.group(2)}")
    return ids


def fetch_kereliuk_filtered() -> set[str]:
    """Скачать union трёх filtered.txt из репо Kereliuk."""
    ids: set[str] = set()
    for fname in KERELIUK_FILES:
        url = KERELIUK_BASE + fname
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        ids |= _parse_kereliuk_file(text)
    if not ids:
        raise RuntimeError(
            "Не удалось распарсить Kereliuk filtered files — пустой union."
        )
    return ids


def _embedded_clean_set(genres_dir: Path) -> set[str]:
    """Fallback: все треки GTZAN минус KNOWN_DUPLICATES ∪ KNOWN_MISLABELS."""
    bad = set(KNOWN_DUPLICATES) | set(KNOWN_MISLABELS) | set(KNOWN_DISTORTIONS)
    clean: set[str] = set()
    for genre_dir in sorted(p for p in genres_dir.iterdir() if p.is_dir()):
        for wav in genre_dir.glob("*.wav"):
            track_id = wav.stem  # e.g. blues.00000
            if track_id not in bad:
                clean.add(track_id)
    return clean


def get_clean_track_ids(
    genres_dir: Path | str,
    method: str = "kereliuk",
    verbose: bool = True,
) -> set[str]:
    """Вернуть set 'чистых' track_id (формат: '<genre>.<5 digits>').

    method:
        "kereliuk"      — скачать filtered.txt из репо Kereliuk (рекомендуется).
        "embedded"      — использовать встроенный список (мин. фильтр).
        "kereliuk_or_embedded" — попробовать Kereliuk, при ошибке fallback.
    """
    genres_dir = Path(genres_dir)

    if method == "kereliuk":
        ids = fetch_kereliuk_filtered()
    elif method == "embedded":
        ids = _embedded_clean_set(genres_dir)
    elif method == "kereliuk_or_embedded":
        try:
            ids = fetch_kereliuk_filtered()
            if verbose:
                print(f"[sturm_filter] Kereliuk OK: {len(ids)} clean tracks")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[sturm_filter] Kereliuk failed ({e}); embedded fallback")
            ids = _embedded_clean_set(genres_dir)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Финальная проверка: оставляем только те, чьи файлы реально существуют.
    existing: set[str] = set()
    for genre_dir in sorted(p for p in genres_dir.iterdir() if p.is_dir()):
        for wav in genre_dir.glob("*.wav"):
            if wav.stem in ids:
                existing.add(wav.stem)

    if verbose:
        print(
            f"[sturm_filter] method={method}: {len(existing)} clean tracks "
            f"(из {sum(1 for _ in genres_dir.rglob('*.wav'))} всего)"
        )
    return existing


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.sturm_filter <genres_dir>")
        sys.exit(1)

    clean = get_clean_track_ids(sys.argv[1], method="kereliuk_or_embedded")
    print(f"Clean tracks: {len(clean)}")
    # Распределение по жанрам
    by_genre: dict[str, int] = {}
    for tid in clean:
        g = tid.split(".")[0]
        by_genre[g] = by_genre.get(g, 0) + 1
    for g in sorted(by_genre):
        print(f"  {g}: {by_genre[g]}")
