# Data

## TL;DR - how to work with this data

We want to train and evaluate sequence-based model, so as the training data
we only need the amino acid sequences and the indices of binding sites (pockets) we want to predict.

- sequences of all the proteins from our dataset can be found in `sequences.csv`.

- `hackathon-data(in).csv` contains indices of the residues of all binding sites.

The difficult part is choosing which of the binding sites we consider *cryptic*.
To make this choice, use metrics saved in `pockets_mean_plddt.csv` and `pockets_rmsd.csv` datasets.

### `sequences.csv`

This dataset contains amino acid sequences of all the unique proteins in
our dataset. The sequences are extracted from AlhpaFold-predicted apo structures
and indexed by the AlphaFold DB IDs (e.g. `AF-P00044-F1-model_v4`).

### `hackathon-data(in).csv`

Data description from the email:

The CSV file contains a list of ligands bound to binding sites. However, we didn’t find any apo structures for these binding sites.

The main idea is to iterate through the binding sites and identify those that may exhibit significant conformational changes. Since we can't compare the binding sites to existing experimentally determined apo structures from the PDB, data from the AlphaFold (AF) database can be used instead. If the pocket region in the AF structure differs significantly, it may suggest that the pocket is indeed flexible.

A few notes on the CSV file:

- Each row contains information about a ligand binding site defined by a holo structure and its ligand, for which no apo structure was found.

- A single PDB structure may appear multiple times, as it can contain multiple ligands.

- The holo structure is specified by the pdb_id and chain columns.

- The ligand is described in the query_poi column, formatted as {ligand-chain-id}_{ligand-id}_{ligand-index} - e.g., K_BES_1003 means the ligand BES is found in chain K at position 1003.

- The pocket selections in both the PDB and AlphaFold structures are provided in the pdb_pocket_selection and alphafold_pocket_selection columns, respectively. These use PyMOL selection syntax, but it should be easy to convert them into any format you need. If the format is not clear, feel free to reach me out.

The indices in pdb_pocket_selection use the auth_seq_id numbering (yes, the PDB has two types of residue numbering, and it can be a real pain to deal with). Feel free to ask - I've struggled with it multiple times, so I might be able to help.

## pLDDT and RMSD

These metrics can tell us which of the predefined pockets can be considered *cryptic*.
The threshold is not universal and we should evaluate different strategies.

### pockets_mean_plddt.csv

For each pocket (as in `alphafold_pocket_selection`) in the AlphaFold-predicted
apo protein structures, this dataset contains the mean pLDDT of its residues.

AlphaFold DB classifies model confidence as follows:
- pLDDT > 90 - very high
- 90 > pLDDT > 70 - high
- 70 > pLDDT > 50 - low
- 50 > pLDDT - very low

**Baseline: consider pockets with mean pLDDT < 70 cryptic.**

### pockets_rmsd.csv

For each pocket and apo-holo conformations pair (apo - AlphaFold-predicted,
holo - experimental from PDB) this dataset contains RMSD between its residues
in both conformations. Note that this dataset is much larger than `pockets_mean_plddt.csv`, because for each apo structure there are multiple corresponding holo structures.

Some pocket residues are not present in experimental holo structures (are *not observed*). `missing_holo_residues` specifies these missing residues. In the case
that only *some* of the pocket's residues are missing, RMSD was calculated ignoring
those residues.

Authors of the CryptoBench paper set the threshold at > 2Å pocket RMSD between
experimental structures.

**Baseline: consider pockets with RMSD > 2 cryptic.**




