from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[1]  # -> project/src → project/

# Data paths
DATA_DIR = ROOT_DIR / "data"
HACKATHON_DATA = DATA_DIR / "hackathon-data(in).csv"

STRUCTURES_DIR = DATA_DIR / "structures"
ALPHAFOLD_STRUCTURES_DIR = STRUCTURES_DIR / "alphafold"
PDB_STRUCTURES_DIR = STRUCTURES_DIR / "pdb"
