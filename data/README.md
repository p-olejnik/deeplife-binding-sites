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