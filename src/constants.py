from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[1]  # -> project/src → project/

# Data paths
DATA_DIR = ROOT_DIR / "data"
HACKATHON_DATA = DATA_DIR / "hackathon-data(in).csv"

# Structures
STRUCTURES_DIR = DATA_DIR / "structures"
ALPHAFOLD_STRUCTURES_DIR = STRUCTURES_DIR / "alphafold"
PDB_STRUCTURES_DIR = STRUCTURES_DIR / "pdb"

# PDB and UniProt structure IDs
ALPHAFOLD_STRUCTURES_IDS = STRUCTURES_DIR / "alphafold_structure_ids.txt"
PDB_STRUCTURES_IDS = STRUCTURES_DIR / "pdb_structure_ids.txt"

# Pocket metrics data
POCKETS_PLDDT = DATA_DIR / "pockets_mean_plddt.csv"
POCKETS_RMSD = DATA_DIR / "pockets_rmsd.csv"

# CryptoBench dataset
CRYPTOBENCH_DIR = DATA_DIR / "cryptobench"
CRYPTOBENCH_DATASET = CRYPTOBENCH_DIR / "dataset.json"
CRYPTOBENCH_FOLDS = CRYPTOBENCH_DIR / "folds.json"

# Sequences (of amino acids)
SEQUENCES = DATA_DIR / "sequences.csv"
CRYPTOBENCH_SEQUENCES = CRYPTOBENCH_DIR / "sequences.csv"

# Our new pockets dataset
DATASET_DIR = DATA_DIR / "dataset"
DATASET = DATASET_DIR / "dataset.json"
NON_CRYPTIC_DATASET = DATASET_DIR / "non_cryptic_dataset.json"
