from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[1]  # -> project/src → project/

# Data paths
DATA_DIR = ROOT_DIR / "data"
HACKATHON_DATA = DATA_DIR / "hackathon-data(in).csv"

STRUCTURES_DIR = DATA_DIR / "structures"
ALPHAFOLD_STRUCTURES_DIR = STRUCTURES_DIR / "alphafold"
PDB_STRUCTURES_DIR = STRUCTURES_DIR / "pdb"

# PDB and UniProt structure IDs
ALPHAFOLD_STRUCTURES_IDS = STRUCTURES_DIR / "alphafold_structure_ids.txt"
PDB_STRUCTURES_IDS = STRUCTURES_DIR / "pdb_structure_ids.txt"

# CryptoBench dataset
CRYPTOBENCH_DIR = DATA_DIR / "cryptobench"
CRYPTOBENCH_DATASET = CRYPTOBENCH_DIR / "dataset.json"
CRYPTOBENCH_FOLDS = CRYPTOBENCH_DIR / "folds.json"

