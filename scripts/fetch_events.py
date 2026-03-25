import json
import requests
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1"
    "/catalog/datasets/evenements-publics-openagenda/records"
)

LOCATION_FILTER = "location_region='Île-de-France'"
# ou : "location_department='974'"  # La Réunion
# ou : "location_city='Paris'"

DATE_START = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
DATE_END   = (datetime.now() + timedelta(days=180)).strftime("%Y-%m-%d")

WHERE_FILTER = (
    f"{LOCATION_FILTER} "
    f"AND firstdate_begin >= '{DATE_START}' "
    f"AND firstdate_begin <= '{DATE_END}'"
)

PAGE_SIZE  = 100   # maximum autorisé par ODS
MAX_EVENTS = 1000

OUTPUT_FILE = Path(__file__).parent.parent / "data" / "raw_events.json"

def fetch_events() -> list[dict]:
    events = []
    offset = 0

    while len(events) < MAX_EVENTS:
        params = {
            "where":  WHERE_FILTER,
            "limit":  PAGE_SIZE,
            "offset": offset,
        }
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        batch = response.json().get("results", [])
        if not batch:
            break

        events.extend(batch)
        offset += PAGE_SIZE
        print(f"{len(events)} événements récupérés...")

    return events[:MAX_EVENTS]


if __name__ == "__main__":
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Récupération des événements...")
    events = fetch_events()

    OUTPUT_FILE.write_text(json.dumps(events, ensure_ascii=False, indent=2))
    print(f"Terminé — {len(events)} événements sauvegardés dans {OUTPUT_FILE}")
