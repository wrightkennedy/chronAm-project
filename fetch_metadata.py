import os
import json
import time
import requests
import pandas as pd
from datetime import datetime
from .config import init_project


def fetch_missing_metadata(
    project_dir: str,
    newspapers: list,
    csv_path: str = None,
    progress_callback=None
) -> str:
    """
    Fetches missing newspaper metadata (latitude/longitude) from Chronicling America.
    - newspapers: list of newspaper titles to update; if empty, auto-detects missing coords.
    - csv_path: path to the existing CSV of newspaper XY info.
    Returns the path to the updated CSV file.
    """
    # Initialize project directories
    cfg = init_project(project_dir)
    data_dir = os.path.join(project_dir, 'data')
    # Paths for intermediate files
    unmatched_json = os.path.join(data_dir, 'unmatched_newspaper_ids.json')
    fetched_csv = os.path.join(data_dir, 'missing_newspaper_metadata.csv')
    # Determine existing & output CSV paths
    if csv_path is None:
        csv_path = cfg['csv']
    updated_csv = os.path.join(data_dir, 'ChronAm_newspapers_XY_updated.csv')

    # Load existing CSV
    df = pd.read_csv(csv_path)
    # Identify rows to fetch
    if newspapers:
        df_to_fetch = df[df['Title'].isin(newspapers)].copy()
    else:
        df_to_fetch = df[df[['Long', 'Lat']].isna().any(axis=1)].copy()

    # Save unmatched list for record
    unique_ids = dict(zip(df_to_fetch['SN'], df_to_fetch['Title']))
    with open(unmatched_json, 'w', encoding='utf-8') as f:
        json.dump(unique_ids, f, ensure_ascii=False, indent=2)

    # Prepare rows for fetched CSV
    fetched_rows = []
    for idx, row in df_to_fetch.iterrows():
        sn = row['SN']
        title = row['Title']
        url = f"https://chroniclingamerica.loc.gov/lccn/{sn}.json"
        try:
            resp = requests.get(url, headers={"User-Agent": "chronam-tool/1.0"})
            resp.raise_for_status()
            data = resp.json()
            # Attempt to extract coordinates if present
            coords = data.get('geometry', {}).get('coordinates', [])
            lon, lat = coords if len(coords) >= 2 else (None, None)
        except Exception:
            lat = None
            lon = None

        # Update main DataFrame
        df.loc[df['SN'] == sn, 'Lat'] = lat
        df.loc[df['SN'] == sn, 'Long'] = lon

        fetched_rows.append({
            'SN': sn,
            'Title': title,
            'Lat': lat,
            'Long': lon,
            'fetched_at': datetime.now().isoformat()
        })

        # Progress callback
        if progress_callback:
            progress_callback(len(fetched_rows))
        # Be polite to the API
        time.sleep(1)

    # Write intermediate fetched CSV
    pd.DataFrame(fetched_rows).to_csv(fetched_csv, index=False)
    # Write updated main CSV
    df.to_csv(updated_csv, index=False)

    return updated_csv
