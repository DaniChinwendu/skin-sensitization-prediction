# Molecular Analysis Results - Column Documentation

## Overview
These CSV files contain molecular analysis results for skin sensitization prediction models. 
Columns are organized into logical groups for clarity.

---

## Column Groups and Descriptions

### 1. IDENTIFIERS
| Column | Description |
|--------|-------------|
| CID | PubChem Compound Identifier - unique numerical ID from PubChem database |
| InChIKey | International Chemical Identifier Key - unique 27-character hash for chemical structures |
| smiles | SMILES notation - Simplified Molecular Input Line Entry System representation |
| IUPAC NAME | International Union of Pure and Applied Chemistry systematic name |

### 2. GROUND TRUTH & EXPERIMENTAL DATA
| Column | Description | Values |
|--------|-------------|--------|
| VALUE | Ground truth skin sensitization status | 1 = Sensitizer, 0 = Non-sensitizer |
| H317 | CLP hazard classification for skin sensitization | 1 = Classified as H317 ("May cause allergic skin reaction"), 0 = Not classified |
| DATATYPE | Source of experimental data | "in vivo" = animal test data (LLNA, GPMT), "IN Chemo" = in chemico data, "prop" = property-based |

**Note on VALUE vs H317:**
- VALUE represents the ground truth label used for model training/testing
- H317 represents the regulatory CLP classification
- These may differ as CLP classification considers multiple evidence sources beyond single assays

### 3. CHEMICAL CLASSIFICATION (ClassyFire)
| Column | Description |
|--------|-------------|
| SUPER CLASS | Highest level chemical taxonomy (e.g., "Benzenoids", "Lipids and lipid-like molecules") |
| CLASS | Mid-level chemical taxonomy (e.g., "Benzene and substituted derivatives") |
| SUBCLASS | Detailed chemical taxonomy (e.g., "Halobenzenes", "Phenols") |

### 4. STRUCTURAL ALERTS (Toxtree)
| Column | Description | Values |
|--------|-------------|--------|
| Alert for Acyl Transfer Agent | Presence of acyl transfer reactive group | 1 = Present, 0 = Absent |
| Alert For Micheal Acceptors | Presence of Michael acceptor reactive group | 1 = Present, 0 = Absent |
| Alert for SN2 | Presence of SN2 (nucleophilic substitution) reactive group | 1 = Present, 0 = Absent |
| Alert for SNAR | Presence of SNAr (aromatic nucleophilic substitution) reactive group | 1 = Present, 0 = Absent |
| Alert for Schiff base | Presence of Schiff base forming group | 1 = Present, 0 = Absent |
| No Skin Sensitization | Toxtree overall prediction | "Yes" = No structural alert triggered (predicted non-sensitizer), "No" = Alert triggered (predicted sensitizer) |
| Number of alert | Total count of Toxtree structural alerts triggered | Integer (0-5) |

### 5. PHYSICOCHEMICAL PROPERTIES
| Column | Description | Units |
|--------|-------------|-------|
| Moleculer_weight | Molecular weight | Daltons (Da) |
| TPSA | Topological Polar Surface Area | Ų |
| OCTANOL_WATER_PARTITION_LOGP_OPERA_PRED | Octanol-water partition coefficient (OPERA predicted) | log units |
| WATER_SOLUBILITY_MOL/L_OPERA_PRED | Water solubility (OPERA predicted) | log(mol/L) |
| VAPOR_PRESSURE_MMHG_OPERA_PRED | Vapor pressure (OPERA predicted) | log(mmHg) |
| BOILING_POINT_DEGC_OPERA_PRED | Boiling point (OPERA predicted) | °C |
| MELTING_POINT_DEGC_OPERA_PRED | Melting point (OPERA predicted) | °C |
| HOMO | Highest Occupied Molecular Orbital energy | eV |
| LUMO | Lowest Unoccupied Molecular Orbital energy | eV |

### 6. MOLECULAR DESCRIPTORS
| Column | Description | Values |
|--------|-------------|--------|
| valid_mol | Valid RDKit molecule object created | True/False |
| total_atoms | Total number of atoms in molecule | Integer |
| has_any_rings | Molecule contains ring structures | True/False |
| total_rings | Total number of rings | Integer |
| has_aromatic_atoms | Molecule contains aromatic atoms | True/False |
| num_aromatic_rings | Number of aromatic rings | Integer |
| aromatic_atoms_count | Count of aromatic atoms | Integer |
| has_conjugated_system | Molecule contains conjugated system | True/False |
| aromatic_conjugation | Has aromatic conjugation | True/False |
| conjugation_type | Type of conjugation system | "No conjugation", "Aromatic only", "Linear only", "Both aromatic and linear" |

### 7. MODEL PREDICTIONS & PERFORMANCE
| Column | Description | Values |
|--------|-------------|--------|
| predictions [MODEL] | Model prediction (column name varies by model) | 1 = Predicted sensitizer, 0 = Predicted non-sensitizer |
| Concordiance | Agreement between model prediction and ground truth | 1 = Prediction matches VALUE, 0 = Prediction differs from VALUE |
| LABEL | Classification outcome based on confusion matrix | TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative |

**LABEL Definitions:**
- **TP (True Positive):** VALUE=1 AND Prediction=1 (Correctly identified sensitizer)
- **TN (True Negative):** VALUE=0 AND Prediction=0 (Correctly identified non-sensitizer)
- **FP (False Positive):** VALUE=0 AND Prediction=1 (Non-sensitizer incorrectly predicted as sensitizer)
- **FN (False Negative):** VALUE=1 AND Prediction=0 (Sensitizer incorrectly predicted as non-sensitizer)

### 8. OTHER
| Column | Description |
|--------|-------------|
| PUBCHEMFP | PubChem fingerprint bit string (881 bits) |

---

## Data Sources
- **Experimental data:** ECHA dossiers, eChemPortal
- **Physicochemical predictions:** OPERA, ChemBCPP, DataWarrior
- **Chemical classification:** ClassyFire
- **Structural alerts:** Toxtree v3.10
- **Molecular descriptors:** RDKit

---

