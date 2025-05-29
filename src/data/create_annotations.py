"""
This scripts creates annotations for cryptic binding sites in protein sequences.
Saves the annotations in a CSV file with columns for UniProt ID,
space-separated, 0-indexed indices of cryptic residues, and sequences.
"""
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np

from src.constants import (
    DATASET,
    SEQUENCES,
    ANNOTATED_SEQUENCES,
)
from src.utils import load_json

def _get_apo_pocket_residues(apo_pocket_selection: list[str]) -> list[int]:
    """
    Extracts the residue indices from the apo pocket selection.

    Args:
        apo_pocket_selection (list[str]): List of apo pocket selections
            in the format "{chain}_{residue_index}" (1-indexed).

    Returns:
        list[int]: List of residue indices (0-indexed).
    """
    return [int(residue.split("_")[-1]) for residue in apo_pocket_selection]


def create_annotations(dataset: dict) -> dict[str, set[int]]:
    """
    Create sequence annotations with binary labels indicating whether
    given residue is a part of a cryptic binding site.

    Args:
        dataset (dict): apo-holo pairs dataset in CryptoBench format.
    Returns:
        dict[str, set[int]]: Dictionary mapping uniprot IDs to sets of
        cryptic residue indices (0-indexed).
    """
    annotated_sequences: dict[str, set[int]] = defaultdict(set)

    for apo_structure in dataset.values():
        for pocket in apo_structure:
            uniprot_id = pocket["uniprot_id"]
            apo_pocket_selection = pocket["apo_pocket_selection"]

            apo_pocket_residues = _get_apo_pocket_residues(apo_pocket_selection)
            for residue in apo_pocket_residues:
                annotated_sequences[uniprot_id].add(residue - 1)

    return annotated_sequences


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Path to the dataset in CryptoBench format.")
    parser.add_argument("--sequences", type=str, default=SEQUENCES,
                        help="Path to the sequences CSV file.")
    parser.add_argument("--output", type=str, default=ANNOTATED_SEQUENCES,
                        help="Path to save the annotated sequences CSV file.")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    dataset = load_json(args.dataset)
    sequences = pd.read_csv(args.sequences, index_col=0)
    sequence_annotations = create_annotations(dataset)

    uniprot_ids = list(sequence_annotations.keys())
    annotations = [" ".join(map(str, sequence_annotations[uniprot_id])) for uniprot_id in uniprot_ids]
    sequences_list = [sequences.loc[uniprot_id]["sequence"] for uniprot_id in uniprot_ids]

    annotations_df = pd.DataFrame({
        "uniprot_id": uniprot_ids,
        "cryptic_residues": annotations,
        "sequence": sequences_list
    })
    annotations_df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
