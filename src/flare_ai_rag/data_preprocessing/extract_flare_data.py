import requests
import json

FLARE_FTSO_URL = "https://api.flare.network/ftso/price"

def fetch_flare_data():
    """
    Fetches real-time FTSO (Flare Time Series Oracle) data.
    """
    response = requests.get(FLARE_FTSO_URL)
    data = response.json()

    with open("processed_data/flare_data.json", "w") as f:
        json.dump(data, f, indent=4)

    print("✅ Flare FTSO data extracted and saved.")

if __name__ == "__main__":
    fetch_flare_data()
