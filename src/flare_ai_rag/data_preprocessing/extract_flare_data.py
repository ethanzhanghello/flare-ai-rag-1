import requests
import json
import os
import structlog

logger = structlog.get_logger(__name__)

FLARE_FTSO_URL = "https://api.flare.network/ftso/price"
OUTPUT_PATH = "processed_data/flare_data.json"

def fetch_flare_data():
    """
    Fetches real-time FTSO (Flare Time Series Oracle) data and stores it in JSON format.
    """
    try:
        logger.info("Fetching Flare FTSO data...", url=FLARE_FTSO_URL)
        response = requests.get(FLARE_FTSO_URL, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

        data = response.json()

        # ✅ Ensure `processed_data/` directory exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        # ✅ Save extracted data
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        logger.info("✅ Flare FTSO data extracted and saved successfully.", file=OUTPUT_PATH)

    except requests.RequestException as e:
        logger.error("⚠️ Failed to fetch Flare data!", error=str(e))

if __name__ == "__main__":
    fetch_flare_data()
