import json
import re
from pathlib import Path

INPUT_FILE  = Path(__file__).parent.parent / "data" / "raw_events.json"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "clean_events.json"


def parse_label_fr(value: str | None) -> str:
    """Extrait le label français d'un champ JSON stringifié (ex: attendancemode, status)."""
    if not value:
        return ""
    try:
        return json.loads(value).get("label", {}).get("fr", "")
    except (json.JSONDecodeError, AttributeError):
        return ""


def _build_address(address: str, postalcode: str, city: str, dept: str = "") -> str:
    """Construit une adresse complète sans doublons (comparaison insensible à la casse)."""
    parts = [address]
    if postalcode and postalcode not in address:
        parts.append(postalcode)
    if city and city.lower() not in address.lower():
        parts.append(city)
    if dept and dept.lower() not in address.lower():
        parts.append(dept)
    return ", ".join(parts)


DEPT_ALIASES: dict[str, str] = {
    "seine-st-denis":  "Seine-Saint-Denis",
    "seine-st.-denis": "Seine-Saint-Denis",
}


def normalize_dept(name: str) -> str:
    """Normalise le nom d'un département vers sa forme canonique."""
    if not name:
        return name
    key = name.strip().lower()
    return DEPT_ALIASES.get(key, name.strip().title())


def normalize_district(district: str, postalcode: str) -> str:
    """Normalise le nom d'un quartier : supprime le préfixe 'Quartier', déduit
    l'arrondissement parisien depuis le code postal si le district vaut 'Paris'."""
    if not district:
        return district

    # Déduction de l'arrondissement parisien
    if district == "Paris" and postalcode and postalcode.startswith("750") and len(postalcode) == 5:
        num = int(postalcode[3:])
        suffix = "1er" if num == 1 else f"{num}e"
        return f"Paris {suffix} Arrondissement"

    # Suppression du préfixe "Quartier (de/du/des/d'/de l') "
    district = re.sub(r"^Quartier\s+(de\s+l[a'\u2019]\s*|du\s+|des\s+|d['\u2019]\s*|de\s+)?", "", district).strip()

    # Normalisation des variantes de "Centre-Ville"
    if district.lower() in ("centre ville", "centre-ville"):
        return "Centre-Ville"

    return district


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
    if not description:
        description = title  # fallback : utiliser le titre si la description est vide
    long_description = strip_html(raw.get("longdescription_fr") or "")
    conditions = (raw.get("conditions_fr") or "").strip()
    keywords = raw.get("keywords_fr") or []

    daterange = (raw.get("daterange_fr") or "").strip()
    firstdate_begin = (raw.get("firstdate_begin") or "").strip()
    lastdate_end    = (raw.get("lastdate_end") or "").strip()

    location_postalcode = (raw.get("location_postalcode") or "")
    age_min = raw.get("age_min")
    age_max_raw = raw.get("age_max")
    age_max = None if (age_max_raw is None or age_max_raw >= 99) else age_max_raw
    accessibility = raw.get("accessibility_label_fr") or []
    attendancemode = parse_label_fr(raw.get("attendancemode"))
    status = parse_label_fr(raw.get("status"))

    location_name = (raw.get("location_name") or "").strip()
    location_address = (raw.get("location_address") or "").strip()
    location_city = (raw.get("location_city") or "").strip().title()
    location_district = normalize_district(
        (raw.get("location_district") or "").strip(),
        location_postalcode,
    )
    location_dept = normalize_dept(raw.get("location_department") or "")
    location_region = (raw.get("location_region") or "").strip()
    coordinates = raw.get("location_coordinates")  # dict {lon, lat} ou None

    url = (raw.get("canonicalurl") or "").strip()

    # Champ texte combiné pour la vectorisation RAG
    details = long_description[len(description):].strip(" .") if long_description.startswith(description) else long_description
    parts = [
        f"Titre : {title}",
        f"Description : {description}" if description else "",
        f"Détails : {details}" if details else "",
        f"Conditions : {conditions}" if conditions else "",
        f"Dates : {daterange}" if daterange else "",
        f"Lieu : {location_name}" if location_name else "",
        f"Adresse : {_build_address(location_address, location_postalcode, location_city, location_dept)}" if location_address else "",
        f"Quartier : {location_district}" if location_district else "",
        f"Âge : {age_min}-{age_max} ans" if age_min is not None and age_max is not None
        else f"Âge : à partir de {age_min} ans" if age_min is not None
        else f"Âge : jusqu'à {age_max} ans" if age_max is not None
        else "",
        f"Accessibilité : {', '.join(accessibility)}" if accessibility else "",
    ]
    text = " | ".join(p for p in parts if p)

    return {
        "uid":              raw.get("uid"),
        "title":            title,
        "description":      description,
        "long_description": long_description,
        "conditions":       conditions,
        "keywords":         keywords,
        "daterange":        daterange,
        "firstdate_begin":  firstdate_begin,
        "lastdate_end":     lastdate_end,
        "location_name":    location_name,
        "location_address": location_address,
        "location_city":     location_city,
        "location_district": location_district,
        "location_postalcode": location_postalcode,
        "location_dept":    location_dept,
        "location_region":  location_region,
        "coordinates":      coordinates,
        "age_min":          age_min,
        "age_max":          age_max,
        "accessibility":    accessibility,
        "attendancemode":   attendancemode,
        "status":           status,
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
