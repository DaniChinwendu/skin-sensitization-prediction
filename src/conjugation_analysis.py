%%writefile src/conjugation_analysis.py
!/usr/bin/env python3

# OPTIMIZING SKIN SENSITIZATION PREDICTION: A COMPARATIVE ANALYSIS OF KNN VS RANDOM FOREST
#
# Authors:
# Daniel C. Ukaegbu, ………, ………, Thomas Hartung, Alexandra Maertens
#
# Affiliation:
# Center for Alternatives to Animal Testing (CAAT),
# Johns Hopkins Bloomberg School of Public Health,
# Baltimore, Maryland 21205, United States
#
# Correspondence:
# Alexandra Maertens
# Email: amaerte1@jhu.edu


"""Conjugation and Aromaticity Analysis
Input:  CSV with a 'smiles' column
Output: 'conjugation_analysis_results.csv' and 'molecular_analysis_results.csv'
"""

import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


# -----------------------------
# Core analysis functions
# -----------------------------
def find_conjugation_chains(mol):
    """Return a list of bond indices that are plausibly conjugated.
    This is a simplified heuristic combining aromatic and double bonds.
    Returns all aromatic/double bonds as a single set (not true path tracing).
    """
    if mol is None:
        return []
    conj_bond_idxs = []
    for b in mol.GetBonds():
        if b.GetIsAromatic() or b.GetBondType() == Chem.BondType.DOUBLE:
            conj_bond_idxs.append(b.GetIdx())
    return [conj_bond_idxs] if conj_bond_idxs else []


def detect_conjugated_systems(smiles):
    """Detect conjugation patterns for a single SMILES string.
    Returns a dict of computed structural features.
    
    Note: These are heuristic features for ML/prediction, not rigorous 
    quantum chemical conjugation definitions.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "valid_mol": False,
            "has_conjugated_system": False,
            "aromatic_conjugation": False,
            "linear_conjugation": False,
            "extended_conjugation": False,
            "conjugated_double_bonds": 0,
            "aromatic_rings": 0,
            "total_rings": 0,
            "conjugation_bond_count": 0,
            "aromatic_atoms_count": 0,
            "non_aromatic_double_bonds": 0,
        }

    aromatic_rings = Descriptors.NumAromaticRings(mol)
    total_rings = Descriptors.RingCount(mol)

    # Aromatic atoms imply conjugation
    aromatic_atoms = [a for a in mol.GetAtoms() if a.GetIsAromatic()]
    aromatic_conjugation = len(aromatic_atoms) > 0

    # Non-aromatic double bonds
    double_bonds = [
        b for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.DOUBLE and not b.GetIsAromatic()
    ]

    # Simple linear-conjugation heuristic: adjacent double bonds sharing an atom
    # (This counts pairs of double bonds with shared atoms, not true alternating paths)
    linear_conjugation = False
    conjugated_double_bonds = 0
    if len(double_bonds) >= 2:
        for i, b1 in enumerate(double_bonds):
            a1 = {b1.GetBeginAtomIdx(), b1.GetEndAtomIdx()}
            for b2 in double_bonds[i+1:]:
                a2 = {b2.GetBeginAtomIdx(), b2.GetEndAtomIdx()}
                if a1 & a2:
                    conjugated_double_bonds += 1
                    linear_conjugation = True

    # Extended conjugation feature: count of aromatic + double bonds (>= 3 for "extended")
    # This is a global count, not a path length
    chains = find_conjugation_chains(mol)
    conjugation_bond_count = max((len(c) for c in chains), default=0)
    extended_conjugation = conjugation_bond_count >= 3

    has_conjugated_system = aromatic_conjugation or linear_conjugation

    return {
        "valid_mol": True,
        "has_conjugated_system": has_conjugated_system,
        "aromatic_conjugation": aromatic_conjugation,
        "linear_conjugation": linear_conjugation,
        "extended_conjugation": extended_conjugation,
        "conjugated_double_bonds": conjugated_double_bonds,
        "aromatic_rings": aromatic_rings,
        "total_rings": total_rings,
        "conjugation_bond_count": conjugation_bond_count,
        "aromatic_atoms_count": len(aromatic_atoms),
        "non_aromatic_double_bonds": len(double_bonds),
    }


def classify_conjugation_type(row):
    """Classify the conjugation category for a row of features."""
    if not row.get("has_conjugated_system", False):
        return "No conjugation"
    if row.get("aromatic_conjugation", False) and row.get("linear_conjugation", False):
        return "Both aromatic and linear"
    if row.get("aromatic_conjugation", False):
        return "Aromatic only"
    if row.get("linear_conjugation", False):
        return "Linear only"
    return "Other"


def analyze_molecular_features(smiles):
    """Compute basic aromaticity/ring features for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "valid_mol": False,
            "has_aromatic_atoms": False,
            "num_aromatic_rings": 0,
            "total_rings": 0,
            "has_any_rings": False,
            "aromatic_atoms_count": 0,
            "total_atoms": 0,
        }
    aromatic_atoms = [a for a in mol.GetAtoms() if a.GetIsAromatic()]
    total_rings = Descriptors.RingCount(mol)
    return {
        "valid_mol": True,
        "has_aromatic_atoms": len(aromatic_atoms) > 0,
        "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
        "total_rings": total_rings,
        "has_any_rings": total_rings > 0,
        "aromatic_atoms_count": len(aromatic_atoms),
        "total_atoms": mol.GetNumAtoms(),
    }


# -----------------------------
# Main routine
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Analyze conjugation and aromatic features from SMILES strings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.csv
  %(prog)s input.csv --output-dir results/
  %(prog)s input.csv --output-prefix exp1_
        """
    )
    parser.add_argument(
        'input_csv',
        help='Input CSV file containing a "smiles" column'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Output directory for results (default: current directory)'
    )
    parser.add_argument(
        '--output-prefix',
        default='',
        help='Prefix for output filenames (default: none)'
    )
    
    args = parser.parse_args()
    
    # Construct output paths
    import os
    conj_output = os.path.join(args.output_dir, f"{args.output_prefix}conjugation_analysis_results.csv")
    mol_output = os.path.join(args.output_dir, f"{args.output_prefix}molecular_analysis_results.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Reading input from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    if "smiles" not in df.columns:
        raise ValueError("Input CSV must contain a 'smiles' column.")

    # =====================================================================
    # CONJUGATION ANALYSIS
    # =====================================================================
    print("\nAnalyzing conjugated systems...")
    conj_features = df["smiles"].apply(detect_conjugated_systems)
    conj_df = pd.DataFrame(conj_features.tolist())
    df_conj = pd.concat([df.reset_index(drop=True), conj_df.reset_index(drop=True)], axis=1)
    df_conj["conjugation_type"] = conj_df.apply(classify_conjugation_type, axis=1)

    # Calculate statistics
    total = len(df_conj)
    valid = int(conj_df["valid_mol"].sum())
    invalid = total - valid
    
    pct = lambda x: 100.0 * x / total if total else 0.0
    
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"Total compounds: {total}")
    print(f"Valid molecules: {valid} ({pct(valid):.1f}%)")
    print(f"Invalid SMILES: {invalid} ({pct(invalid):.1f}%)")

    # Conjugation statistics
    total_conj = int(conj_df["has_conjugated_system"].sum())
    aromatic_conj = int(conj_df["aromatic_conjugation"].sum())
    linear_conj = int(conj_df["linear_conjugation"].sum())
    extended_conj = int(conj_df["extended_conjugation"].sum())
    
    print("\n" + "="*60)
    print("CONJUGATION FEATURES")
    print("="*60)
    print(f"Compounds with conjugated systems: {total_conj} ({pct(total_conj):.1f}%)")
    print(f"  - Aromatic conjugation: {aromatic_conj} ({pct(aromatic_conj):.1f}%)")
    print(f"  - Linear conjugation: {linear_conj} ({pct(linear_conj):.1f}%)")
    print(f"  - Extended conjugation (≥3 bonds): {extended_conj} ({pct(extended_conj):.1f}%)")

    # Conjugation type distribution
    print("\n" + "="*60)
    print("CONJUGATION TYPE DISTRIBUTION")
    print("="*60)
    type_counts = df_conj["conjugation_type"].value_counts()
    for conjugation_type, count in type_counts.items():
        print(f"{conjugation_type}: {count} ({pct(count):.1f}%)")

    # Ring statistics
    aromatic_rings_pos = int((conj_df["aromatic_rings"] > 0).sum())
    total_rings_pos = int((conj_df["total_rings"] > 0).sum())
    
    print("\n" + "="*60)
    print("RING ANALYSIS")
    print("="*60)
    print(f"Compounds with aromatic rings: {aromatic_rings_pos} ({pct(aromatic_rings_pos):.1f}%)")
    print(f"Compounds with any rings: {total_rings_pos} ({pct(total_rings_pos):.1f}%)")
    
    # Aromatic ring distribution
    print("\nDistribution of aromatic rings:")
    for rings, count in conj_df["aromatic_rings"].value_counts().sort_index().items():
        print(f"  {rings} ring(s): {count} compounds ({pct(count):.1f}%)")
    
    # Total ring distribution
    print("\nDistribution of total rings:")
    for rings, count in conj_df["total_rings"].value_counts().sort_index().items():
        print(f"  {rings} ring(s): {count} compounds ({pct(count):.1f}%)")

    # Save conjugation results
    df_conj.to_csv(conj_output, index=False)
    print(f"\n{'='*60}")
    print(f"Conjugation analysis results saved to: {conj_output}")
    print(f"{'='*60}")

    # =====================================================================
    # MOLECULAR FEATURES ANALYSIS
    # =====================================================================
    print("\nAnalyzing molecular features...")
    features = df["smiles"].apply(analyze_molecular_features)
    feat_df = pd.DataFrame(features.tolist())
    df_feat = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    # Aromatic statistics
    aromatic_compounds = int(feat_df["has_aromatic_atoms"].sum())
    ring_compounds = int(feat_df["has_any_rings"].sum())
    aromatic_and_rings = int((feat_df['has_aromatic_atoms'] & feat_df['has_any_rings']).sum())
    aliphatic_rings_only = int((feat_df["has_any_rings"] & ~feat_df["has_aromatic_atoms"]).sum())
    
    print("\n" + "="*60)
    print("AROMATICITY ANALYSIS")
    print("="*60)
    print(f"Compounds with aromatic atoms: {aromatic_compounds} ({pct(aromatic_compounds):.1f}%)")
    print(f"Compounds with rings: {ring_compounds} ({pct(ring_compounds):.1f}%)")
    print(f"  - Aromatic rings: {aromatic_and_rings} ({pct(aromatic_and_rings):.1f}%)")
    print(f"  - Aliphatic rings only: {aliphatic_rings_only} ({pct(aliphatic_rings_only):.1f}%)")

    # Save molecular features results
    df_feat.to_csv(mol_output, index=False)
    print(f"\n{'='*60}")
    print(f"Molecular features results saved to: {mol_output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
