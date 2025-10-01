# Project structure:
#
# chronam_project/
# ├── chronam/                      # Python package
# │   ├── __init__.py
# │   ├── config.py                 # central path configuration
# │   ├── download.py               # streaming download logic
# │   ├── merge.py                  # geojson merging
# │   ├── fetch_metadata.py         # missing newspaper metadata
# │   ├── collocate.py              # collocation analysis
# │   └── visualize.py              # plotting routines
# ├── app.py                        # PyQt GUI, imports chronam.* modules
# ├── requirements.txt
# ├── chronam/resources/            # packaged reference data
# │   └── ChronAm_newspapers_XY.csv # default locations table
# └── data/                         # created under project root
#     ├── raw/                      # raw JSON downloads
#     └── processed/                # merged geojson outputs

import os
from pathlib import Path

try:  # Python 3.9+
    from importlib.resources import files
except ImportError:  # pragma: no cover
    files = None

# Default Hugging Face dataset parameters
DATASET_NAME = "dell-research-harvard/AmericanStories"
DATASET_CONFIG_NAME = "subset_years"
DATASET_SPLIT = "train"

# Default file names
DEFAULT_CSV_FILENAME = "ChronAm_newspapers_XY.csv"
DEFAULT_MERGED_GEOJSON = "merged.geojson"
DEFAULT_COLLATED_GEOJSON = "collocated.geojson"


CSV_RESOURCE_PACKAGE = "chronam.resources"
PACKAGE_ROOT = Path(__file__).resolve().parent
_BUNDLED_CSV = PACKAGE_ROOT / "resources" / DEFAULT_CSV_FILENAME


def default_csv_path() -> str:
    """Return the packaged ChronAm CSV path, falling back to local resources."""
    if _BUNDLED_CSV.exists():
        return str(_BUNDLED_CSV)
    if files:
        try:
            resource = files(CSV_RESOURCE_PACKAGE) / DEFAULT_CSV_FILENAME
            return str(resource)
        except ModuleNotFoundError:
            pass
    return str(_BUNDLED_CSV)


def init_project(project_dir: str) -> dict:
    """
    Ensure the project directory structure exists and returns key paths.

    Creates:
      project_dir/data/raw/
      project_dir/data/processed/
    """
    data_dir = os.path.join(project_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    proc_dir = os.path.join(data_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    paths = {
        "project": project_dir,
        "data": data_dir,
        "raw": raw_dir,
        "processed": proc_dir,
        "csv": default_csv_path(),
        "merged_geojson": os.path.join(proc_dir, DEFAULT_MERGED_GEOJSON),
        "collocated_geojson": os.path.join(proc_dir, DEFAULT_COLLATED_GEOJSON)
    }
    return paths
