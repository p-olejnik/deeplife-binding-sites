import argparse
from collections import defaultdict
import re

import pandas as pd
from tqdm import tqdm

from src.constants import (
    HACKATHON_DATA,
    POCKETS_PLDDT,
    POCKETS_RMSD,
    DATASET,
    NON_CRYPTIC_DATASET,
)
from src.utils import save_json


PLDDT_THRESHOLD = 70.0
RMSD_THRESHOLD = 2.0


def parse_pocket_selection(
    selection: str,
    default_chain: str = "A",
) -> tuple[str, list[int]]:
    """
    Parse a PyMOL selection string to extract chain ID and residue indices.
    """
    chain_match = re.search(r"chain\s+([A-Za-z])", selection)
    chain_id = chain_match.group(1) if chain_match else default_chain

    resi_match = re.search(r"resi\s+([0-9+\s]+)", selection)
    if not resi_match:
        raise ValueError("No residue information found in selection string.")

    resi_list = [
        int(r.strip())
        for r in resi_match.group(1).split("+")
        if r.strip().isdigit()
    ]

    return chain_id, resi_list


def parse_ligand_id(query_poi: str) -> tuple[str, str, str]:
    """
    Parse ligand infor string formatted as {ligand_chain}_{ligand}_{ligand_index}
    """
    parts = query_poi.split("_")
    if len(parts) != 3:
        raise ValueError(
            "Ligand information must be in the format 'chain_ligand_index'."
        )

    ligand_chain, ligand, ligand_index = parts
    return ligand_chain, ligand, ligand_index


def parse_pocket_data(row):
    apo_structure_name = row["alphafold_pocket_selection"].split(" ")[0]

    uniprot_id = row["uniprot_id_rmsd"]
    holo_pdb_id = row["pdb_id"]
    holo_chain = row["chain"]
    apo_chain = "A"  # AlphaFold structures have only one chain

    ligand_chain, ligand, ligand_index = parse_ligand_id(row["query_poi"])

    apo_pymol_selection = row["alphafold_pocket_selection"]
    holo_pymol_selection = row["pdb_pocket_selection"]

    apo_pocket_chain, apo_pocket_residues = parse_pocket_selection(
        apo_pymol_selection, default_chain=apo_chain
    )
    holo_pocket_chain, holo_pocket_residues = parse_pocket_selection(
        holo_pymol_selection, default_chain=holo_chain
    )

    apo_pocket_selection = [
        apo_pocket_chain + "_" + str(residue)
        for residue in apo_pocket_residues
    ]
    holo_pocket_selection = [
        holo_pocket_chain + "_" + str(residue)
        for residue in holo_pocket_residues
    ]

    pRMSD = row["pocket_rmsd"]
    pLDDT = row["mean_plddt"]  # This key is not used in the original dataset
    is_main_holo_structure = False # TODO what is main holo structure?

    pocket_data = {
        "uniprot_id": uniprot_id,
        "holo_pdb_id": holo_pdb_id,
        "holo_chain": holo_chain,
        "apo_chain": apo_chain,
        "ligand_chain": ligand_chain,
        "ligand": ligand,
        "ligand_index": ligand_index,
        "apo_pocket_selection": apo_pocket_selection,
        "holo_pocket_selection": holo_pocket_selection,
        "pRMSD": pRMSD,
        "pLDDT": pLDDT,
        "is_main_holo_structure": is_main_holo_structure,
    }

    return apo_structure_name, pocket_data


def create_dataset(
    pockets_data: pd.DataFrame,
    ignore_missing_residues: bool = False,
    verbose: bool = False,
) -> tuple[dict, dict]:
    """
    Create dataset in CryptoBench format from pockets data.
    Returns a tuple of two dictionaries: cryptic and non-cryptic pockets.
    """
    cryptic_pockets = defaultdict(list)
    non_cryptic_pockets = defaultdict(list)

    for _, row in tqdm(
        pockets_data.iterrows(),
        total=len(pockets_data),
        desc="Processing pockets",
    ):
        apo_structure_name, pocket_data = parse_pocket_data(row)

        if ignore_missing_residues and pd.notna(row["missing_holo_residues"]):
            if verbose:
                print(f"Skipping {apo_structure_name} due to missing residues: {row['missing_holo_residues']}")
            continue

        if row["is_cryptic"]:
            cryptic_pockets[apo_structure_name].append(pocket_data)
        else:
            non_cryptic_pockets[apo_structure_name].append(pocket_data)

    return cryptic_pockets, non_cryptic_pockets


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plddt_threshold",
        type=float,
        default=PLDDT_THRESHOLD,
        help="Consider pockets with mean pLDDT below this threshold cryptic.",
    )
    parser.add_argument(
        "--rmsd_threshold",
        type=float,
        default=RMSD_THRESHOLD,
        help="Consider pockets with RMSD above this threshold cryptic.",
    )
    parser.add_argument(
        "--ignore_missing_residues",
        action="store_true",
        help="If True, ignore pockets with missing residues in the selection.",
    )
    parser.add_argument(
        "--save_non_cryptic",
        action="store_true",
        help="Save non-cryptic pockets to a separate file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional information during processing.",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    hackathon_data = pd.read_csv(HACKATHON_DATA)
    plddt_data = pd.read_csv(POCKETS_PLDDT)
    rmsd_data = pd.read_csv(POCKETS_RMSD)

    # Merge pLDDT and RMSD data
    metrics_data = pd.merge(
        plddt_data,
        rmsd_data,
        on=["alphafold_pocket_selection"],
        suffixes=("_plddt", "_rmsd"),
    )

    # Drop pockets with missing pLDDT or RMSD values
    metrics_data = metrics_data.dropna(subset=["mean_plddt", "pocket_rmsd"])

    # Filter cryptic pockets
    metrics_data["is_cryptic"] = (
        metrics_data["mean_plddt"] < args.plddt_threshold
    ) & (metrics_data["pocket_rmsd"] > args.rmsd_threshold)

    cryptic_dataset, non_cryptic_dataset = create_dataset(
        metrics_data, args.ignore_missing_residues, args.verbose
    )

    # Save datasets
    save_json(cryptic_dataset, DATASET)
    print(f"Cryptic pockets dataset saved to {DATASET}")

    if args.save_non_cryptic:
        save_json(non_cryptic_dataset, NON_CRYPTIC_DATASET)
        print(f"Non-cryptic pockets dataset saved to {NON_CRYPTIC_DATASET}")


if __name__ == "__main__":
    main()
