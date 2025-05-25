import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Structure import Structure
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src.constants import HACKATHON_DATA, ALPHAFOLD_STRUCTURES_DIR, PDB_STRUCTURES_DIR


N_WORKERS = max(mp.cpu_count() - 1, 1)                       # leave one core free


def parse_pocket_selection(
    selection: str,
    default_chain: str = "A",
) -> tuple[str, list[int]]:
    """
    Parse a PyMOL selection string to extract chain ID and residue indices.
    """
    chain_match = re.search(r'chain\s+([A-Za-z])', selection)
    chain_id = chain_match.group(1) if chain_match else default_chain

    resi_match = re.search(r'resi\s+([0-9+\s]+)', selection)
    if not resi_match:
        raise ValueError("No residue information found in selection string.")

    resi_list = [int(r.strip()) for r in resi_match.group(1).split('+') if r.strip().isdigit()]

    return chain_id, resi_list


def extract_plddt(structure: Structure) -> list[tuple[int, float]] | list[float]:
    """
    Extract pLDDT values from a PDB structure file.

    Returns:
        A list of tuples, where each tuple contains the residue index and its
        corresponding pLDDT value.
    """
    # AlphaFold predictions contain one model and one chain
    model = next(structure.get_models())
    chain = next(model.get_chains())

    residue_plddt_scores = []

    for residue in chain:
        atoms = list(residue.get_atoms())
        # All atoms in a residue a have the same pLDDT in AF outputs, so
        # we get the pLDDT value from the B-factor field of the first atom
        plddt = atoms[0].get_bfactor()
        residue_plddt_scores.append(plddt)

    return residue_plddt_scores


def residues_mean_plddt(residues: list[int], plddt_scores: list[float]) -> float:
    """
    Calculate the mean pLDDT score for a list of residues.
    """
    selected_scores = [plddt_scores[r - 1] for r in residues] # residues are 1-indexed
    return sum(selected_scores) / len(selected_scores)




def _worker(idx_row):
    """
    Stand-alone function executed in a separate process.
    Returns (original_idx, mean_plddt or None).
    """
    idx, row = idx_row
    uniprot_id = row['uniprot_id']
    selection_str = row['alphafold_pocket_selection']
    pdb_parser = PDBParser(QUIET=True)

    # Parse residue selection
    try:
        _, residues = parse_pocket_selection(selection_str)
    except Exception as e:
        print(f"[worker] parse error for {idx}, {uniprot_id}: {e}")
        return idx, None

    # Read structure file
    structure_file = ALPHAFOLD_STRUCTURES_DIR / f"{uniprot_id}.cif"
    if not structure_file.exists():
        print(f"Warning: PDB file for {uniprot_id} not found. Skipping.")
        return idx, None

    try:
        structure = pdb_parser.get_structure(uniprot_id, structure_file)
        plddt_scores = extract_plddt(structure)
        mean_plddt = residues_mean_plddt(residues, plddt_scores)
        return idx, mean_plddt
    except Exception as e:
        print(f"Error calculating mean pLDDT for {idx}, {uniprot_id}: {e}")
        return idx, None


def parallel_mean_plddt(df, n_workers=N_WORKERS, chunksize=32):
    """
    Wrapper that submits jobs and re-orders results so they match df.index.
    """
    print(f"Using {n_workers} workers for parallel processing.")
    results = [None] * len(df)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_worker, item): item[0]
            for item in df.iterrows()
        }

        for fut in tqdm(as_completed(futures), total=len(futures)):
            idx, mean_val = fut.result()
            results[idx] = mean_val   # index is the original integer position

    return results


def main():
    data = pd.read_csv(HACKATHON_DATA)
    pockets_plddt_df = data[['uniprot_id', 'alphafold_pocket_selection']].drop_duplicates(ignore_index=True)
    pocket_mean_plddt = parallel_mean_plddt(pockets_plddt_df)
    pockets_plddt_df['mean_plddt'] = pocket_mean_plddt
    pockets_plddt_df.to_csv("pockets_mean_plddt.csv", index=False)
    print("Mean pLDDT values calculated and saved to pockets_mean_plddt.csv")


if __name__ == "__main__":
    main()