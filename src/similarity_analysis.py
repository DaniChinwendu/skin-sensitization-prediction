%%writefile src/similarity_analysis.py
#!/usr/bin/env python3

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

"""
Tanimoto Similarity Matrix and Pairwise Analysis
Computes fingerprint similarity matrices and extracts similar pairs.
Supports Morgan, RDKit, AtomPair, MACCS Keys, and PubChem fingerprints.
"""

import argparse
import os
from base64 import b64decode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.DataStructs import ExplicitBitVect


def decode_pubchem_fp(pcfp_base64):
    """Decode PubChem fingerprint from base64 string.
    
    Args:
        pcfp_base64: Base64-encoded PubChem fingerprint string
    
    Returns:
        Binary string of fingerprint (881 bits)
    """
    try:
        # Decode base64 and convert to binary string
        # PubChem FP: skip first 32 bits (header), use bits 32-913 (881 bits total)
        binary_str = "".join(["{:08b}".format(x) for x in b64decode(pcfp_base64)])[32:913]
        return binary_str
    except Exception as e:
        print(f"  Warning: Failed to decode PubChem FP: {e}")
        return None


def convert_bitstring_to_bitvect(bitstring):
    """Convert binary string to RDKit ExplicitBitVect for similarity calculations.
    
    Args:
        bitstring: Binary string (e.g., "01101...")
    
    Returns:
        RDKit ExplicitBitVect object
    """
    if bitstring is None:
        return None
    
    n_bits = len(bitstring)
    bitvect = ExplicitBitVect(n_bits)
    
    # Set bits that are 1
    for i, bit in enumerate(bitstring):
        if bit == '1':
            bitvect.SetBit(i)
    
    return bitvect


def generate_fingerprints(df, fp_type='morgan', radius=3, n_bits=512, pubchem_column=None):
    """Generate molecular fingerprints from SMILES strings or decode PubChem FPs.
    
    Args:
        df: DataFrame with 'smiles' column (and optionally PubChem FP column)
        fp_type: Type of fingerprint ('morgan', 'rdkit', 'atompair', 'maccs', 'pubchem')
        radius: Radius for Morgan fingerprints
        n_bits: Number of bits in fingerprint (not used for MACCS or PubChem)
        pubchem_column: Column name containing base64-encoded PubChem fingerprints
    
    Returns:
        DataFrame with added 'Molecule' and fingerprint columns
    """
    print(f"\nGenerating {fp_type} fingerprints...")
    
    df = df.copy()
    
    # Handle PubChem fingerprints separately (don't need molecules)
    if fp_type == 'pubchem':
        if pubchem_column is None:
            raise ValueError("--pubchem-column must be specified when using --fp-type pubchem")
        
        if pubchem_column not in df.columns:
            raise ValueError(f"Column '{pubchem_column}' not found in input CSV")
        
        print(f"  Decoding PubChem fingerprints from column: {pubchem_column}")
        
        # Decode base64 fingerprints
        df['PubChem_BitString'] = df[pubchem_column].astype(str).apply(decode_pubchem_fp)
        
        # Convert to RDKit BitVect
        df['Fingerprint'] = df['PubChem_BitString'].apply(convert_bitstring_to_bitvect)
        
        valid_count = df['Fingerprint'].notna().sum()
        invalid_count = df['Fingerprint'].isna().sum()
        
        print(f"  Valid fingerprints: {valid_count}")
        print(f"  Invalid fingerprints: {invalid_count}")
        print(f"  Fingerprint size: 881 bits (PubChem standard)")
        
        df_valid = df[df['Fingerprint'].notna()].copy()
        return df_valid
    
    # For other fingerprint types, generate molecules from SMILES
    df['Molecule'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
    
    # Count valid/invalid molecules
    valid_count = df['Molecule'].notna().sum()
    invalid_count = df['Molecule'].isna().sum()
    print(f"  Valid molecules: {valid_count}")
    print(f"  Invalid SMILES: {invalid_count}")
    
    # Generate fingerprints based on type
    if fp_type == 'morgan':
        df['Fingerprint'] = df['Molecule'].apply(
            lambda x: AllChem.GetMorganFingerprintAsBitVect(x, nBits=n_bits, radius=radius, useFeatures=True) 
            if x is not None else None
        )
        print(f"  Fingerprint size: {n_bits} bits (radius={radius})")
    
    elif fp_type == 'rdkit':
        df['Fingerprint'] = df['Molecule'].apply(
            lambda x: Chem.RDKFingerprint(x, fpSize=n_bits) if x is not None else None
        )
        print(f"  Fingerprint size: {n_bits} bits")
    
    elif fp_type == 'atompair':
        df['Fingerprint'] = df['Molecule'].apply(
            lambda x: AllChem.GetHashedAtomPairFingerprintAsBitVect(x, nBits=n_bits) 
            if x is not None else None
        )
        print(f"  Fingerprint size: {n_bits} bits")
    
    elif fp_type == 'maccs':
        df['Fingerprint'] = df['Molecule'].apply(
            lambda x: MACCSkeys.GenMACCSKeys(x) if x is not None else None
        )
        print(f"  Fingerprint size: 167 bits (fixed MACCS Keys)")
    
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}. Choose from: morgan, rdkit, atompair, maccs, pubchem")
    
    # Remove rows with invalid fingerprints
    df_valid = df[df['Fingerprint'].notna()].copy()
    print(f"  Fingerprints generated: {len(df_valid)}")
    
    return df_valid


def compute_tanimoto_matrix(df, id_column='CID'):
    """Compute Tanimoto similarity matrix for fingerprints.
    
    Uses efficient upper-triangle computation to avoid redundant calculations.
    
    Args:
        df: DataFrame with 'Fingerprint' column
        id_column: Column name to use for indexing
    
    Returns:
        DataFrame with similarity matrix
    """
    print("\nComputing Tanimoto similarity matrix...")
    
    fingerprints = df['Fingerprint'].tolist()
    ids = df[id_column].tolist()
    n = len(fingerprints)
    
    # Initialize matrix
    sim_matrix = np.zeros((n, n))
    
    # Fill diagonal
    np.fill_diagonal(sim_matrix, 1.0)
    
    # Compute upper triangle only (more efficient)
    for i in range(n):
        if i % 100 == 0 and i > 0:
            print(f"  Processing row {i}/{n}...")
        
        # Use BulkTanimotoSimilarity for vectorized computation
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i+1:])
        
        # Fill upper triangle
        for j, sim in enumerate(similarities, start=i+1):
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim  # Mirror to lower triangle
    
    print("  Similarity matrix computed")
    
    # Create DataFrame with IDs as index/columns
    sim_df = pd.DataFrame(sim_matrix, index=ids, columns=ids)
    
    return sim_df


def flatten_similarity_matrix(sim_df, threshold=0.7):
    """Convert similarity matrix to pairwise long format.
    
    Extracts only upper triangle to avoid duplicate pairs.
    
    Args:
        sim_df: Similarity matrix DataFrame
        threshold: Minimum similarity to include
    
    Returns:
        DataFrame with columns ['Chemical_x', 'Chemical_y', 'Similarity']
    """
    print(f"\nExtracting pairs with similarity >= {threshold}...")
    
    pairs = []
    ids = sim_df.index.tolist()
    
    # Iterate only over upper triangle to avoid duplicates
    for i_idx, i in enumerate(ids):
        for j in ids[i_idx + 1:]:  # Only pairs where j > i
            similarity = sim_df.loc[i, j]
            if similarity >= threshold:
                pairs.append((i, j, similarity))
    
    pair_df = pd.DataFrame(pairs, columns=['Chemical_x', 'Chemical_y', 'Similarity'])
    pair_df = pair_df.sort_values('Similarity', ascending=False).reset_index(drop=True)
    
    print(f"  Found {len(pair_df)} pairs above threshold")
    
    return pair_df


def plot_similarity_distribution(sim_df, output_file=None):
    """Plot histogram of similarity score distribution.
    
    Args:
        sim_df: Similarity matrix DataFrame
        output_file: Optional path to save plot
    """
    print("\nGenerating similarity distribution plot...")
    
    # Extract upper triangle values only (avoid diagonal and duplicates)
    sim_values = sim_df.values[np.triu_indices_from(sim_df.values, k=1)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(sim_values, bins=50, edgecolor='k', alpha=0.7, color='steelblue')
    plt.xlabel('Tanimoto Similarity', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Tanimoto Similarity Scores', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_sim = np.mean(sim_values)
    median_sim = np.median(sim_values)
    plt.axvline(mean_sim, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.3f}')
    plt.axvline(median_sim, color='g', linestyle='--', linewidth=2, label=f'Median: {median_sim:.3f}')
    plt.legend()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def analyze_similarity(input_csv, id_column='CID', fp_type='morgan', radius=3, n_bits=512, 
                      threshold=0.7, output_dir='.', output_prefix='', pubchem_column=None):
    """Complete similarity analysis pipeline.
    
    Args:
        input_csv: Path to CSV with 'smiles' column (or PubChem FP column)
        id_column: Column name for compound IDs
        fp_type: Fingerprint type ('morgan', 'rdkit', 'atompair', 'maccs', 'pubchem')
        radius: Morgan fingerprint radius
        n_bits: Fingerprint size (not used for MACCS or PubChem)
        threshold: Similarity threshold for pairs
        output_dir: Output directory
        output_prefix: Prefix for output files
        pubchem_column: Column name for base64-encoded PubChem fingerprints
    """
    print("="*60)
    print("TANIMOTO SIMILARITY ANALYSIS")
    print("="*60)
    print(f"Input: {input_csv}")
    
    if fp_type == 'pubchem':
        print(f"Fingerprint: {fp_type} (881 bits, from column '{pubchem_column}')")
    elif fp_type == 'maccs':
        print(f"Fingerprint: {fp_type} (167 bits fixed)")
    elif fp_type == 'morgan':
        print(f"Fingerprint: {fp_type} (radius={radius}, bits={n_bits})")
    else:
        print(f"Fingerprint: {fp_type} (bits={n_bits})")
    
    print(f"Threshold: {threshold}")
    
    # Load data
    df = pd.read_csv(input_csv)
    
    # Validate columns
    if fp_type != 'pubchem' and 'smiles' not in df.columns:
        raise ValueError("Input CSV must contain a 'smiles' column (unless using PubChem fingerprints)")
    
    if id_column not in df.columns:
        raise ValueError(f"Input CSV must contain a '{id_column}' column")
    
    print(f"\nLoaded {len(df)} compounds")
    
    # Generate fingerprints
    df = generate_fingerprints(df, fp_type=fp_type, radius=radius, n_bits=n_bits, pubchem_column=pubchem_column)
    
    if len(df) == 0:
        print("Error: No valid fingerprints to analyze")
        return
    
    # Compute similarity matrix
    sim_df = compute_tanimoto_matrix(df, id_column=id_column)
    
    # Save similarity matrix
    matrix_file = os.path.join(output_dir, f"{output_prefix}similarity_matrix_{fp_type}.csv")
    sim_df.to_csv(matrix_file)
    print(f"\nSimilarity matrix saved to: {matrix_file}")
    
    # Extract similar pairs
    pair_df = flatten_similarity_matrix(sim_df, threshold=threshold)
    
    if len(pair_df) > 0:
        pairs_file = os.path.join(output_dir, f"{output_prefix}similarity_pairs_{fp_type}.csv")
        pair_df.to_csv(pairs_file, index=False)
        print(f"Similar pairs saved to: {pairs_file}")
        
        # Display top pairs
        print(f"\nTop 10 most similar pairs:")
        print(pair_df.head(10).to_string(index=False))
    else:
        print(f"\nNo pairs found with similarity >= {threshold}")
    
    # Plot distribution
    plot_file = os.path.join(output_dir, f"{output_prefix}similarity_distribution_{fp_type}.png")
    plot_similarity_distribution(sim_df, output_file=plot_file)
    
    # Summary statistics
    sim_values = sim_df.values[np.triu_indices_from(sim_df.values, k=1)]
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total pairwise comparisons: {len(sim_values)}")
    print(f"Mean similarity: {np.mean(sim_values):.3f}")
    print(f"Median similarity: {np.median(sim_values):.3f}")
    print(f"Min similarity: {np.min(sim_values):.3f}")
    print(f"Max similarity: {np.max(sim_values):.3f}")
    print(f"Std deviation: {np.std(sim_values):.3f}")
    print(f"Pairs above threshold ({threshold}): {len(pair_df)}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compute Tanimoto similarity matrix and extract similar pairs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Morgan fingerprint (default)
  %(prog)s input.csv
  
  # MACCS Keys fingerprint
  %(prog)s input.csv --fp-type maccs
  
  # PubChem fingerprint (from pre-computed column)
  %(prog)s input.csv --fp-type pubchem --pubchem-column PUBCHEMFP
  
  # Custom Morgan parameters
  %(prog)s input.csv --fp-type morgan --radius 2 --n-bits 1024 --threshold 0.8
  
  # AtomPair fingerprint
  %(prog)s input.csv --fp-type atompair --n-bits 2048
  
  # Custom ID column and output
  %(prog)s input.csv --id-column compound_id --output-dir results/ --output-prefix exp1_

Fingerprint Types:
  morgan    - Morgan (circular) fingerprints with customizable radius and size
  maccs     - MACCS Keys (167 fixed structural keys)
  pubchem   - PubChem fingerprints (881 bits, requires base64-encoded column)
  rdkit     - RDKit path-based fingerprints
  atompair  - Atom pair fingerprints
        """
    )
    
    parser.add_argument(
        'input_csv',
        help='Input CSV file with "smiles" column (or PubChem FP column)'
    )
    parser.add_argument(
        '--id-column',
        default='CID',
        help='Column name for compound IDs (default: CID)'
    )
    parser.add_argument(
        '--fp-type',
        choices=['morgan', 'rdkit', 'atompair', 'maccs', 'pubchem'],
        default='morgan',
        help='Fingerprint type (default: morgan)'
    )
    parser.add_argument(
        '--pubchem-column',
        help='Column name containing base64-encoded PubChem fingerprints (required for --fp-type pubchem)'
    )
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Morgan fingerprint radius (default: 3, ignored for other types)'
    )
    parser.add_argument(
        '--n-bits',
        type=int,
        default=512,
        help='Fingerprint size in bits (default: 512, ignored for MACCS/PubChem)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Similarity threshold for extracting pairs (default: 0.7)'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Output directory (default: current directory)'
    )
    parser.add_argument(
        '--output-prefix',
        default='',
        help='Prefix for output filenames (default: none)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis
    analyze_similarity(
        input_csv=args.input_csv,
        id_column=args.id_column,
        fp_type=args.fp_type,
        radius=args.radius,
        n_bits=args.n_bits,
        threshold=args.threshold,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        pubchem_column=args.pubchem_column
    )


if __name__ == "__main__":
    main()
