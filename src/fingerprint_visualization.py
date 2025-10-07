%%writefile src/fingerprint_visualization.py
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
RDKit Fingerprint Bit Visualization
Generates Morgan and AtomPair fingerprints and visualizes key bits.
Saves PNG images of highlighted substructures for each active bit.
"""

import argparse
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D


def generate_morgan_fingerprint(mol, radius=3, fp_size=512):
    """Generate Morgan fingerprint with bit info for visualization.
    
    Args:
        mol: RDKit molecule object
        radius: Morgan fingerprint radius
        fp_size: Fingerprint size in bits
    
    Returns:
        tuple: (fingerprint, bitinfo dictionary)
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    
    fp = gen.GetFingerprint(mol, additionalOutput=ao)
    bitinfo = ao.GetBitInfoMap()
    
    return fp, bitinfo


def generate_atompair_fingerprint(mol, fp_size=512):
    """Generate AtomPair fingerprint with bit info for visualization.
    
    Args:
        mol: RDKit molecule object
        fp_size: Fingerprint size in bits
    
    Returns:
        tuple: (fingerprint, bitinfo dictionary)
    """
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fp_size)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    
    fp = gen.GetFingerprint(mol, additionalOutput=ao)
    bitinfo = ao.GetBitInfoMap()
    
    return fp, bitinfo


def visualize_morgan_bits(mol, smiles, fp, bitinfo, output_dir, max_bits=10):
    """Visualize and save Morgan fingerprint bits.
    
    Args:
        mol: RDKit molecule object
        smiles: SMILES string (for filename)
        fp: Morgan fingerprint
        bitinfo: Bit information dictionary
        output_dir: Directory to save images
        max_bits: Maximum number of bits to visualize
    """
    on_bits = list(fp.GetOnBits())
    visualizable_bits = [bit for bit in on_bits if bit in bitinfo]
    
    print(f"\nMorgan Fingerprint Analysis:")
    print(f"  SMILES: {smiles}")
    print(f"  Total bits set: {len(on_bits)}")
    print(f"  Visualizable bits: {len(visualizable_bits)}")
    
    if not visualizable_bits:
        print("  Warning: No visualizable bits found")
        return
    
    # Limit number of bits to visualize
    bits_to_visualize = visualizable_bits[:max_bits]
    print(f"  Visualizing {len(bits_to_visualize)} bits...")
    
    for bit in bits_to_visualize:
        try:
            # Create safe filename
            safe_smiles = smiles.replace('/', '_').replace('\\', '_')[:30]
            filename = os.path.join(output_dir, f"morgan_bit_{bit}_{safe_smiles}.png")
            
            # Draw and save the bit
            img = Draw.DrawMorganBit(mol, bit, bitinfo)
            img.save(filename)
            print(f"    Saved: {filename}")
        except Exception as e:
            print(f"    Error visualizing bit {bit}: {e}")


def visualize_atompair_bits(mol, smiles, fp, bitinfo, output_dir, max_bits=10):
    """Visualize and save AtomPair fingerprint bits.
    
    Since RDKit doesn't have a DrawAtomPairBit helper like DrawMorganBit,
    we manually highlight the atom pairs from bitinfo.
    
    Args:
        mol: RDKit molecule object
        smiles: SMILES string (for filename)
        fp: AtomPair fingerprint
        bitinfo: Bit information dictionary
        output_dir: Directory to save images
        max_bits: Maximum number of bits to visualize
    """
    on_bits = list(fp.GetOnBits())
    visualizable_bits = [bit for bit in on_bits if bit in bitinfo]
    
    print(f"\nAtomPair Fingerprint Analysis:")
    print(f"  SMILES: {smiles}")
    print(f"  Total bits set: {len(on_bits)}")
    print(f"  Visualizable bits: {len(visualizable_bits)}")
    
    if not visualizable_bits:
        print("  Warning: No visualizable bits found")
        return
    
    # Limit number of bits to visualize
    bits_to_visualize = visualizable_bits[:max_bits]
    print(f"  Visualizing {len(bits_to_visualize)} bits...")
    
    for bit in bits_to_visualize:
        try:
            # Get atom pairs for this bit
            atom_pairs = bitinfo[bit]
            
            # Collect all atoms involved in these pairs
            highlight_atoms = []
            highlight_bonds = []
            
            for pair_info in atom_pairs:
                atom1, atom2 = pair_info
                highlight_atoms.extend([atom1, atom2])
                
                # Find bond between atoms if it exists
                bond = mol.GetBondBetweenAtoms(atom1, atom2)
                if bond is not None:
                    highlight_bonds.append(bond.GetIdx())
            
            # Remove duplicates
            highlight_atoms = list(set(highlight_atoms))
            highlight_bonds = list(set(highlight_bonds))
            
            # Create safe filename
            safe_smiles = smiles.replace('/', '_').replace('\\', '_')[:30]
            filename = os.path.join(output_dir, f"atompair_bit_{bit}_{safe_smiles}.png")
            
            # Draw molecule with highlights
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightBonds=highlight_bonds
            )
            drawer.FinishDrawing()
            
            # Save image
            with open(filename, 'wb') as f:
                f.write(drawer.GetDrawingText())
            
            print(f"    Saved: {filename}")
        except Exception as e:
            print(f"    Error visualizing bit {bit}: {e}")


def analyze_fingerprints(smiles, fp_type, output_dir, radius=3, fp_size=512, max_bits=10):
    """Generate and visualize fingerprints for a SMILES string.
    
    Args:
        smiles: SMILES string
        fp_type: 'morgan', 'atompair', or 'both'
        output_dir: Directory to save images
        radius: Morgan fingerprint radius
        fp_size: Fingerprint size in bits
        max_bits: Maximum number of bits to visualize per fingerprint
    """
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: Invalid SMILES '{smiles}'")
        return
    
    print("="*60)
    print(f"Analyzing molecule: {smiles}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Generate and visualize Morgan fingerprint
    if fp_type in ['morgan', 'both']:
        morgan_fp, morgan_bitinfo = generate_morgan_fingerprint(mol, radius, fp_size)
        visualize_morgan_bits(mol, smiles, morgan_fp, morgan_bitinfo, output_dir, max_bits)
    
    # Generate and visualize AtomPair fingerprint
    if fp_type in ['atompair', 'both']:
        atompair_fp, atompair_bitinfo = generate_atompair_fingerprint(mol, fp_size)
        visualize_atompair_bits(mol, smiles, atompair_fp, atompair_bitinfo, output_dir, max_bits)
    
    print("\n" + "="*60)
    print("Analysis complete")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate and visualize RDKit fingerprint bits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --smiles "CCOCCl" --fp morgan
  %(prog)s --smiles "c1ccccc1O" --fp both --out-dir results/
  %(prog)s --smiles "CCOCCl" --fp atompair --fp-size 1024 --max-bits 5
        """
    )
    
    parser.add_argument(
        '--smiles',
        required=True,
        help='SMILES string to analyze'
    )
    parser.add_argument(
        '--fp',
        choices=['morgan', 'atompair', 'both'],
        default='both',
        help='Fingerprint type to generate (default: both)'
    )
    parser.add_argument(
        '--out-dir',
        default='fingerprint_bits',
        help='Output directory for bit visualizations (default: fingerprint_bits)'
    )
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Morgan fingerprint radius (default: 3)'
    )
    parser.add_argument(
        '--fp-size',
        type=int,
        default=512,
        help='Fingerprint size in bits (default: 512)'
    )
    parser.add_argument(
        '--max-bits',
        type=int,
        default=10,
        help='Maximum number of bits to visualize (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Run analysis
    analyze_fingerprints(
        smiles=args.smiles,
        fp_type=args.fp,
        output_dir=args.out_dir,
        radius=args.radius,
        fp_size=args.fp_size,
        max_bits=args.max_bits
    )


if __name__ == "__main__":
    main()
