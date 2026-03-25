import json
import re
from pathlib import Path

INPUT_FILE  = Path(__file__).parent.parent / "data" / "raw_events.json"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "clean_events.json"


def strip_html(text: str) -> str:
    """Supprime les balises HTML et normalise les espaces."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_event(raw: dict) -> dict | None:
    """
    Extrait et nettoie les champs utiles d'un événement brut.
    Retourne None si les champs obligatoires sont absents.
    """
    title = (raw.get("title_fr") or "").strip()
    if not title:
        return None

    description      = (raw.get("description_fr") or "").strip()
    long_description = strip_html(raw.get("longdescription_fr") or "")
    conditions       = (raw.get("conditions_fr") or "").strip()
    keywords         = raw.get("keywords_fr") or []

    date_begin = raw.get("firstdate_begin") or ""
    date_end   = raw.get("lastdate_end") or ""
    daterange  = (raw.get("daterange_fr") or "").strip()

    location_name    = (raw.get("location_name") or "").strip()
    location_address = (raw.get("location_address") or "").strip()
    location_city    = (raw.get("location_city") or "").strip()
    location_dept    = (raw.get("location_department") or "").strip()
    location_region  = (raw.get("location_region") or "").strip()
    coordinates      = raw.get("location_coordinates")  # dict {lon, lat} ou None

    url = (raw.get("canonicalurl") or "").strip()

    # Champ texte combiné pour la vectorisation RAG
    parts = [
        f"Titre : {title}",
        f"Description : {description}" if description else "",
        f"Détails : {long_description}" if long_description else "",
        f"Conditions : {conditions}" if conditions else "",
        f"Mots-clés : {', '.join(keywords)}" if keywords else "",
        f"Dates : {daterange}" if daterange else f"Début : {date_begin}",
        f"Lieu : {location_name}" if location_name else "",
        f"Adresse : {location_address}" if location_address else "",
        f"Ville : {location_city}" if location_city else "",
        f"Département : {location_dept}" if location_dept else "",
        f"Région : {location_region}" if location_region else "",
    ]
    text = " | ".join(p for p in parts if p)

    return {
        "uid":              raw.get("uid"),
        "title":            title,
        "description":      description,
        "long_description": long_description,
        "conditions":       conditions,
        "keywords":         keywords,
        "date_begin":       date_begin,
        "date_end":         date_end,
        "daterange":        daterange,
        "location_name":    location_name,
        "location_address": location_address,
        "location_city":    location_city,
        "location_dept":    location_dept,
        "location_region":  location_region,
        "coordinates":      coordinates,
        "url":              url,
        "text":             text,
    }


def clean_events(raw_events: list[dict]) -> list[dict]:
    cleaned = []
    seen_uids: set[str] = set()
    skipped_missing = 0
    skipped_duplicate = 0

    for raw in raw_events:
        event = clean_event(raw)

        if event is None:
            skipped_missing += 1
            continue

        uid = event["uid"]
        if uid in seen_uids:
            skipped_duplicate += 1
            continue

        seen_uids.add(uid)
        cleaned.append(event)

    print(f"  Evenements conserves   : {len(cleaned)}")
    print(f"  Ignores (titre absent) : {skipped_missing}")
    print(f"  Ignores (doublons)     : {skipped_duplicate}")
    return cleaned


if __name__ == "__main__":
    print(f"Lecture de {INPUT_FILE}...")
    raw_events = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    print(f"{len(raw_events)} evenements bruts charges.")

    print("Nettoyage en cours...")
    clean = clean_events(raw_events)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Termine — {len(clean)} evenements sauvegardes dans {OUTPUT_FILE}")
