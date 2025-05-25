import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from pathlib import Path

from Bio.PDB import Superimposer, PDBParser, MMCIFParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Structure import Structure
import pandas as pd
from tqdm import tqdm

from src.constants import (
    HACKATHON_DATA,
    ALPHAFOLD_STRUCTURES_DIR,
    PDB_STRUCTURES_DIR,
)


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


def get_pocket_atoms(
    structure: Structure,
    chain_id: str,
    pocket_residues: list[int],
    atom_names=["CA"],
    strict=False,
) -> tuple[dict[int, list[Atom]], list[int]]:
    """
    Extract atoms from a specified pocket in a PDB structure.

    Args:
        structure: BioPython Structure object
        chain_id: Chain identifier (e.g., 'A')
        pocket_residues: List of residue numbers
        atom_names: List of atom names to extract (default: ['CA'])
        strict: If True, raise error for missing residues/atoms

    Returns:
        List of Atom objects

    Raises:
        ValueError: If strict=True and residues/atoms are missing
        KeyError: If chain_id doesn't exist
    """
    pocket_atoms = {residue_id: [] for residue_id in pocket_residues}
    missing_residues = []
    missing_atoms = []

    try:
        model = structure[0]
        chain = model[chain_id]
    except KeyError:
        available_chains = [c.id for c in model.get_chains()]
        raise KeyError(f"Chain '{chain_id}' not found. Available chains: {available_chains}")

    for res_id in pocket_residues:
        if res_id not in chain:
            missing_residues.append(res_id)
            if strict:
                raise ValueError(f"Residue {res_id} not found in chain {chain_id}")
            continue

        residue = chain[res_id]
        atoms_found_for_residue = []

        for atom_name in atom_names:
            if atom_name in residue:
                pocket_atoms[res_id].append(residue[atom_name])
                atoms_found_for_residue.append(atom_name)
            else:
                missing_atoms.append((res_id, atom_name))
                if strict:
                    available_atoms = [atom.id for atom in residue.get_atoms()]
                    raise ValueError(
                        f"Atom '{atom_name}' not found in residue {res_id}. "
                        f"Available atoms: {available_atoms}"
                    )

        # If no atoms found for this residue, log it
        if not atoms_found_for_residue and not strict:
            missing_atoms.append((res_id, "all requested atoms"))

    # Report issues if not in strict mode
    if missing_residues:
        print(f"Warning: Missing residues in chain {chain_id}: {missing_residues}")

    if missing_atoms:
        print(f"Warning: Missing atoms: {missing_atoms[:5]}{'...' if len(missing_atoms) > 5 else ''}")

    return pocket_atoms, missing_residues


def calculate_pocket_rmsd(
    holo_struct: Structure,
    apo_struct: Structure,
    holo_chain: str,
    apo_chain: str,
    holo_residues: list[int],
    apo_residues: list[int],
) -> tuple[float | None, list[int], list[int]]:
    """
    Calculate the RMSD between two pockets in holo and apo structures.
    This function assumes that the residues in both pockets are aligned by their
    C-alpha atoms.
    """
    # Return dicts of [residue_id: [Atom, ...]] for each pocket
    holo_atoms, missing_holo_residues = get_pocket_atoms(holo_struct, holo_chain, holo_residues)
    apo_atoms, missing_apo_residues = get_pocket_atoms(apo_struct, apo_chain, apo_residues)

    holo_atoms_list = []
    apo_atoms_list = []
    for holo_res_id, apo_res_id in zip(holo_residues, apo_residues):
        # Check if both residues have atoms
        if len(holo_atoms[holo_res_id]) == 0 or len(apo_atoms[apo_res_id]) == 0:
            continue

        holo_atoms_list.append(holo_atoms[holo_res_id][0])  # Use first atom (CA)
        apo_atoms_list.append(apo_atoms[apo_res_id][0])

    if len(holo_atoms_list) == 0 or len(apo_atoms_list) == 0:
        print(f"Warning: No atoms found for RMSD calculation between {holo_chain} and {apo_chain}.")
        return None, missing_holo_residues, missing_apo_residues

    superimposer = Superimposer()
    superimposer.set_atoms(holo_atoms_list, apo_atoms_list)
    return superimposer.rms, missing_holo_residues, missing_apo_residues


def process_single_rmsd(row_data):
    """
    Process a single RMSD calculation - designed for multiprocessing.

    Args:
        row_data: tuple of (index, row_dict, pdb_structures_dir, alphafold_structures_dir)

    Returns:
        tuple: (index, rmsd_value, uniprot_id, pdb_id, error_message)
    """
    idx, row, pdb_dir, alphafold_dir = row_data

    try:
        pdb_id = row["pdb_id"]
        uniprot_id = row["uniprot_id"]
        holo_pocket_selection = row["pdb_pocket_selection"]
        apo_pocket_selection = row["alphafold_pocket_selection"]

        if pd.isna(holo_pocket_selection) or pd.isna(apo_pocket_selection):
            return idx, None, [], [], uniprot_id, pdb_id, "Missing pocket selection"

        # Parse pocket selections
        holo_chain, holo_residues = parse_pocket_selection(holo_pocket_selection)
        apo_chain, apo_residues = parse_pocket_selection(apo_pocket_selection)

        # Validate residue lists
        assert len(apo_residues) == len(holo_residues), (
            f"Residue count mismatch for {uniprot_id}-{pdb_id}: "
            f"{len(apo_residues)} vs {len(holo_residues)}"
        )

        # Check file existence
        holo_file = pdb_dir / f"{pdb_id}.cif"
        apo_file = alphafold_dir / f"{uniprot_id}.cif"

        if not holo_file.exists() or not apo_file.exists():
            return idx, None, [], [], uniprot_id, pdb_id, "Structure files not found"

        # Load structures
        pdb_parser = PDBParser(QUIET=True)
        cif_parser = MMCIFParser(QUIET=True)

        holo_structure = cif_parser.get_structure(pdb_id, holo_file)
        apo_structure = pdb_parser.get_structure(uniprot_id, apo_file)

        # Calculate RMSD
        (
            rmsd,
            missing_holo_residues,
            missing_apo_residues,
        ) = calculate_pocket_rmsd(
            holo_structure,
            apo_structure,
            holo_chain,
            apo_chain,
            holo_residues,
            apo_residues,
        )

        return idx, rmsd, missing_holo_residues, missing_apo_residues, uniprot_id, pdb_id, None

    except Exception as e:
        return (
            idx,
            None,
            [],
            [],
            row.get("uniprot_id", ""),
            row.get("pdb_id", ""),
            str(e),
        )


def parallel_rmsd_calculation(data, n_workers=None, chunk_size=10):
    """
    Calculate RMSDs in parallel using multiprocessing.

    Args:
        data: DataFrame with structure and pocket information
        n_workers: Number of worker processes (default: CPU count - 1)
        chunk_size: Size of chunks to process at once

    Returns:
        list: RMSD values in the same order as input data
    """
    if n_workers is None:
        n_workers = max(mp.cpu_count() - 1, 1)

    print(f"Using {n_workers} workers for parallel RMSD calculation")

    # Prepare data for workers
    work_items = [
        (idx, row.to_dict(), PDB_STRUCTURES_DIR, ALPHAFOLD_STRUCTURES_DIR)
        for idx, row in data.iterrows()
    ]

    # Initialize results array
    results = [None] * len(data)
    errors = []

    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_idx = {
            executor.submit(process_single_rmsd, item): item[0]
            for item in work_items
        }

        # Collect results with progress bar
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Calculating RMSDs",
            ascii=True,
        ):
            try:
                idx, rmsd, missing_holo_residues, missing_apo_residues, uniprot_id, pdb_id, error_msg = future.result()
                results[idx] = (rmsd, missing_holo_residues, missing_apo_residues)

                if error_msg:
                    errors.append(
                        f"Row {idx} ({uniprot_id}-{pdb_id}): {error_msg}"
                    )

            except Exception as e:
                idx = future_to_idx[future]
                results[idx] = (None, [], [])
                errors.append(f"Row {idx}: Unexpected error - {str(e)}")

    # Print error summary
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    return results


def process_dataset(data: pd.DataFrame, workers: int | None = None, verbose: bool = True) -> pd.DataFrame:
    rmsd_output = parallel_rmsd_calculation(data, n_workers=workers)

    # Create results DataFrame
    valid_data = data.copy()
    pocket_rmsd, missing_holo, missing_apo = zip(*rmsd_output)
    valid_data["pocket_rmsd"] = pocket_rmsd
    valid_data["missing_holo_residues"] = [','.join(map(str, x)) for x in missing_holo]
    valid_data["missing_apo_residues"] = [','.join(map(str, x)) for x in missing_apo]

    # Print statistics
    if verbose:
        successful_calculations = sum(1 for x in pocket_rmsd if x is not None)
        print(f"\nResults:")
        print(f"  Total rows processed: {len(data)}")
        print(f"  Successful RMSD calculations: {successful_calculations}")
        print(f"  Failed calculations: {len(data) - successful_calculations}")

        if successful_calculations > 0:
            valid_rmsds = [x for x in pocket_rmsd if x is not None]
            print(f"  RMSD statistics:")
            print(f"    Mean: {sum(valid_rmsds) / len(valid_rmsds):.3f} Å")
            print(f"    Min: {min(valid_rmsds):.3f} Å")
            print(f"    Max: {max(valid_rmsds):.3f} Å")

    return valid_data


def main():
    print("Loading data...")
    data = pd.read_csv(HACKATHON_DATA)

    # Divide data into batches for processing
    batch_size = 10000
    batches = [data[i:i + batch_size].reset_index(drop=True) for i in range(0, len(data), batch_size)]
    print(f"Processing {len(batches)} batches of size {batch_size}...")

    results = []

    for i, batch in enumerate(batches):
        print(f"Processing batch {i + 1}/{len(batches)}...")

        # Check if batch already processed
        output_file = f"pockets_rmsd_batch_{i + 1}.csv"
        if Path(output_file).exists():
            print(f"Batch {i + 1} already processed. Loading results from {output_file}.")
            valid_data = pd.read_csv(output_file)
        else:
            print(f"Calculating RMSD for batch {i + 1}...")
            # Process the batch
            valid_data = process_dataset(batch)
            valid_data.to_csv(output_file, index=False)
            print(f"Batch {i + 1} processed successfully.")

        results.append(valid_data)

    # Combine all results
    final_results = pd.concat(results, ignore_index=True)
    final_results.to_csv("pockets_rmsd.csv", index=False)

    print("All batches processed successfully.")
    print("Final results saved to pockets_rmsd.csv")

if __name__ == "__main__":
    main()
