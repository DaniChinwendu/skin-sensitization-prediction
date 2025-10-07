%%writefile src/train_evaluate.py
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

"""
Training and Testing Script for Skin Sensitization Prediction
Supports multiple fingerprint types: MACCS, Avalon, RDKit AtomPair, PubChem
Uses original CV and SHAP methodology for consistency with paper results
"""

import argparse
import os
import pandas as pd
import numpy as np
import seaborn as sns  # FIXED: was 'sn'
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

from base64 import b64decode

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_fscore_support,
                             classification_report)

from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdMolDescriptors
from rdkit.Avalon import pyAvalonTools


# --- Fingerprint Decoding and Generation ---

def decode_pubchem_fp(pcfp_base64):
    """Decode PubChem fingerprint from base64 string (881 bits)."""
    try:
        return "".join(["{:08b}".format(x) for x in b64decode(pcfp_base64)])[32:913]
    except:
        return None


def convert_bitstring_to_numpy(bitstring):
    """Convert binary string to numpy array."""
    if bitstring is None:
        return None
    return np.array([int(b) for b in bitstring])


def generate_maccs_fp(mol):
    """Generate MACCS keys fingerprint (167 bits)."""
    if mol is None:
        return None
    return np.array(MACCSkeys.GenMACCSKeys(mol))


def generate_avalon_fp(mol, n_bits=512):
    """Generate Avalon fingerprint."""
    if mol is None:
        return None
    return np.array(pyAvalonTools.GetAvalonFP(mol, nBits=n_bits))


def generate_atompair_fp(mol, n_bits=512):
    """Generate RDKit AtomPair fingerprint."""
    if mol is None:
        return None
    return np.array(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits))


# --- Feature Preparation ---

def prepare_fingerprints(df, fp_type='maccs', n_bits=512, pubchem_column=None):
    """Generate fingerprints and prepare feature matrix.

    Args:
        df: DataFrame with 'smiles' column
        fp_type: Type of fingerprint ('maccs', 'avalon', 'atompair', 'pubchem')
        n_bits: Number of bits for Avalon/AtomPair
        pubchem_column: Column name for PubChem fingerprints if using pre-computed

    Returns:
        DataFrame with fingerprint features
    """
    print(f"\nPreparing {fp_type.upper()} fingerprints...")

    if fp_type == 'pubchem':
        if pubchem_column is None or pubchem_column not in df.columns:
            raise ValueError(f"PubChem fingerprints require column '{pubchem_column}'")

        # Decode PubChem fingerprints
        df['FP_BitString'] = df[pubchem_column].astype(str).apply(decode_pubchem_fp)
        df['FP_Array'] = df['FP_BitString'].apply(convert_bitstring_to_numpy)
        n_features = 881
    else:
        # Generate molecules
        df['Molecule'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)

        # Generate fingerprints based on type
        if fp_type == 'maccs':
            df['FP_Array'] = df['Molecule'].apply(generate_maccs_fp)
            n_features = 167
        elif fp_type == 'avalon':
            df['FP_Array'] = df['Molecule'].apply(lambda m: generate_avalon_fp(m, n_bits))
            n_features = n_bits
        elif fp_type == 'atompair':
            df['FP_Array'] = df['Molecule'].apply(lambda m: generate_atompair_fp(m, n_bits))
            n_features = n_bits
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")

    # Convert to DataFrame
    valid_fps = df['FP_Array'].notna()
    print(f"  Valid fingerprints: {valid_fps.sum()}/{len(df)}")

    fp_list = df.loc[valid_fps, 'FP_Array'].tolist()
    fp_df = pd.DataFrame(fp_list, index=df[valid_fps].index)
    fp_df.columns = [f'{i}_MF' for i in range(n_features)]  # Match original naming

    return fp_df


def prepare_features(df, fp_df, feature_columns):
    """Combine fingerprints with additional molecular descriptors.

    Args:
        df: Original DataFrame
        fp_df: Fingerprint DataFrame
        feature_columns: List of additional feature column names to include

    Returns:
        Combined feature DataFrame
    """
    # Start with fingerprints
    features = fp_df.copy()

    # Add additional features if they exist
    for col in feature_columns:
        if col in df.columns:
            features[col] = df.loc[features.index, col].values
        else:
            print(f"  Warning: Column '{col}' not found, skipping")

    return features


# --- Model Training and Evaluation ---

def train_and_evaluate(X_train, y_train, X_test, y_test, model_type='rf', **model_params):
    """Train model and evaluate on test set.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_type: 'rf' for RandomForest or 'knn' for KNeighbors
        **model_params: Model hyperparameters

    Returns:
        Trained model and predictions
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*60}")

    # Create model
    if model_type == 'rf':
        model = RandomForestClassifier(**model_params)
    elif model_type == 'knn':
        model = KNeighborsClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    print(f"Training on {len(X_train)} samples...")
    model.fit(X_train, y_train.values.ravel())

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    print("\nTest Set Performance:")
    print_metrics(y_test, y_pred)

    return model, y_pred


def print_metrics(y_true, y_pred):
    """Print classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def plot_confusion_matrix(y_true, y_pred, title, output_file=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f"Confusion Matrix - {title}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {output_file}")
    else:
        plt.show()

    plt.close()


# --- Cross-Validation (Original Method) ---

def perform_cross_validation(train_fp, test_fp, model_type='rf', n_splits=10, fp_size=167, **model_params):
    """Perform cross-validation using original methodology.

    Concatenates train and test, uses only fingerprint features for CV.
    This matches the original paper methodology.

    Args:
        train_fp: Training feature DataFrame
        test_fp: Test feature DataFrame
        model_type: 'rf' or 'knn'
        n_splits: Number of CV folds
        fp_size: Number of fingerprint columns
        **model_params: Model hyperparameters

    Returns:
        CV scores and predictions
    """
    print(f"\n{'='*60}")
    print(f"{n_splits}-Fold Cross-Validation ({model_type.upper()})")
    print(f"{'='*60}")

    # Concatenate train and test
    cv_data = pd.concat([train_fp, test_fp], axis=0)

    # Use only fingerprint columns (first fp_size columns)
    X_cv = cv_data.iloc[:, :fp_size]
    y_cv = cv_data['VALUE']

    print(f"  Total samples: {len(X_cv)}")
    print(f"  Features used: {fp_size} (fingerprint only)")
    print(f"  Class distribution: {dict(y_cv.value_counts())}")

    # Create model
    if model_type == 'rf':
        model = RandomForestClassifier(**model_params)
    elif model_type == 'knn':
        model = KNeighborsClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # K-Fold CV (original method)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Cross-validation scores
    print("\nComputing CV scores...")
    fold_scores = cross_val_score(model, X_cv, y_cv, scoring='f1', cv=kf)

    print(f"\nF1 Scores per fold:")
    for i, score in enumerate(fold_scores, 1):
        print(f"  Fold {i}: {score:.4f}")

    print(f"\nCV F1 Mean: {fold_scores.mean():.4f}, Std: {fold_scores.std():.4f}")

    # Get predictions for detailed metrics
    print("\nComputing CV predictions...")
    y_cv_pred = cross_val_predict(model, X_cv, y_cv, cv=kf)

    # Detailed performance
    prec, rec, f1, _ = precision_recall_fscore_support(y_cv, y_cv_pred, average='macro')
    acc = np.mean(y_cv == y_cv_pred)

    print("\nOverall CV Performance:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    return fold_scores, y_cv_pred


# --- SHAP Analysis (Original Method) ---

def perform_shap_analysis(model, X_test, output_file=None, max_display=20):
    """Perform SHAP analysis using original methodology.

    Args:
        model: Trained Random Forest model
        X_test: Test features
        output_file: Path to save SHAP plot
        max_display: Number of top features to display
    """
    print(f"\n{'='*60}")
    print("SHAP Feature Importance Analysis")
    print(f"{'='*60}")

    if not isinstance(model, RandomForestClassifier):
        print("  SHAP analysis only supported for Random Forest")
        return

    print("Computing SHAP values...")

    # Use shap.Explainer (original method)
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    # For binary classification, use class 1 (positive class)
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    else:
        shap_values_plot = shap_values

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_plot, X_test, max_display=max_display, show=False)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nSHAP plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


# --- Main Pipeline ---

def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate skin sensitization prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with MACCS fingerprints
#   %(prog)s train.csv test.csv

  # Use Avalon fingerprints
#   %(prog)s train.csv test.csv --fp-type avalon --n-bits 1024

  # Use PubChem fingerprints
#   %(prog)s train.csv test.csv --fp-type pubchem --pubchem-column PUBCHEMFP

  # Train both KNN and RF
#   %(prog)s train.csv test.csv --models knn rf

  # Custom output directory
#   %(prog)s train.csv test.csv --output-dir results/
        """
    )

    parser.add_argument('train_csv', help='Training data CSV')
    parser.add_argument('test_csv', help='Test data CSV')
    parser.add_argument('--fp-type', choices=['maccs', 'avalon', 'atompair', 'pubchem'],
                       default='maccs', help='Fingerprint type (default: maccs)')
    parser.add_argument('--n-bits', type=int, default=512,
                       help='Number of bits for Avalon/AtomPair (default: 512)')
    parser.add_argument('--pubchem-column', default='PUBCHEMFP',
                       help='Column name for PubChem fingerprints (default: PUBCHEMFP)')
    parser.add_argument('--label-column', default='VALUE',
                       help='Column name for labels (default: VALUE)')
    parser.add_argument('--additional-features', nargs='*',
                       default=['Alert for Acyl Transfer Agent', 'Alert For Micheal Acceptors',
                               'Alert for SN2', 'Alert for SNAR', 'Alert for Schiff base',
                               'OCTANOL_WATER_PARTITION_LOGP_OPERA_PRED',
                               'VAPOR_PRESSURE_MMHG_OPERA_PRED',
                               'WATER_SOLUBILITY_MOL/L_OPERA_PRED'],
                       help='Additional feature columns to include')
    parser.add_argument('--models', nargs='+', choices=['knn', 'rf'], default=['knn', 'rf'],
                       help='Models to train (default: both)')
    parser.add_argument('--cv-folds', type=int, default=10,
                       help='Number of CV folds (default: 10)')
    parser.add_argument('--no-scaling', action='store_true',
                       help='Skip MinMaxScaler (use if features already scaled)')
    parser.add_argument('--no-rounding', action='store_true',
                       help='Skip rounding after scaling (recommended for better performance)')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("SKIN SENSITIZATION PREDICTION")
    print("="*60)
    print(f"Training data: {args.train_csv}")
    print(f"Test data: {args.test_csv}")
    print(f"Fingerprint: {args.fp_type.upper()}")
    print(f"Models: {', '.join(args.models).upper()}")

    # Load data
    print("\nLoading data...")
    df_train = pd.read_csv(args.train_csv)
    df_test = pd.read_csv(args.test_csv)

    print(f"  Training samples: {len(df_train)}")
    print(f"  Test samples: {len(df_test)}")

    # Validate label column
    if args.label_column not in df_train.columns or args.label_column not in df_test.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in data")

    # Prepare fingerprints
    train_fp_only = prepare_fingerprints(df_train, args.fp_type, args.n_bits, args.pubchem_column)
    test_fp_only = prepare_fingerprints(df_test, args.fp_type, args.n_bits, args.pubchem_column)

    # Determine fingerprint size
    fp_size = train_fp_only.shape[1]
    print(f"\nFingerprint size: {fp_size} bits")

    # Add additional features
    train_fp = prepare_features(df_train, train_fp_only, args.additional_features)
    test_fp = prepare_features(df_test, test_fp_only, args.additional_features)

    # Add labels to feature DataFrames (needed for CV)
    train_fp[args.label_column] = df_train.loc[train_fp.index, args.label_column].values
    test_fp[args.label_column] = df_test.loc[test_fp.index, args.label_column].values

    print(f"Total features (with descriptors): {train_fp.shape[1] - 1}")  # -1 for label
    print(f"Class distribution (train): {dict(train_fp[args.label_column].value_counts())}")
    print(f"Class distribution (test): {dict(test_fp[args.label_column].value_counts())}")

    # Prepare train/test splits for modeling (with scaling if requested)
    if not args.no_scaling:
        print("\nApplying MinMaxScaler...")
        scaler_X = MinMaxScaler().fit(train_fp.iloc[:, :fp_size])
        scaler_y = MinMaxScaler().fit(train_fp[[args.label_column]])

        X_train = pd.DataFrame(
            scaler_X.transform(train_fp.iloc[:, :fp_size]),
            columns=train_fp.columns[:fp_size],
            index=train_fp.index
        )
        X_test = pd.DataFrame(
            scaler_X.transform(test_fp.iloc[:, :fp_size]),
            columns=test_fp.columns[:fp_size],
            index=test_fp.index
        )

        y_train = pd.DataFrame(
            scaler_y.transform(train_fp[[args.label_column]]),
            columns=[args.label_column],
            index=train_fp.index
        )
        y_test = pd.DataFrame(
            scaler_y.transform(test_fp[[args.label_column]]),
            columns=[args.label_column],
            index=test_fp.index
        )

        # Apply rounding if not disabled
        if not args.no_rounding:
            print("  Rounding scaled values (use --no-rounding to disable)")
            X_train = X_train.round(0)
            X_test = X_test.round(0)
            y_train = y_train.round(0)
            y_test = y_test.round(0)
    else:
        print("\nSkipping scaling (--no-scaling flag)")
        X_train = train_fp.iloc[:, :fp_size]
        X_test = test_fp.iloc[:, :fp_size]
        y_train = train_fp[[args.label_column]]
        y_test = test_fp[[args.label_column]]

    # Train models
    models_trained = {}

    if 'knn' in args.models:
        model_knn, pred_knn = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            model_type='knn',
            n_neighbors=7,
            weights='distance',
            algorithm='brute',
            metric='euclidean'  # Original metric
        )
        models_trained['knn'] = (model_knn, pred_knn)

        # Plot confusion matrix
        plot_confusion_matrix(
            y_test, pred_knn, "KNN",
            os.path.join(args.output_dir, f"confusion_matrix_knn_{args.fp_type}.png")
        )

    if 'rf' in args.models:
        model_rf, pred_rf = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            model_type='rf',
            n_estimators=55,
            random_state=args.random_state
        )
        models_trained['rf'] = (model_rf, pred_rf)

        # Plot confusion matrix
        plot_confusion_matrix(
            y_test, pred_rf, "Random Forest",
            os.path.join(args.output_dir, f"confusion_matrix_rf_{args.fp_type}.png")
        )

        # SHAP analysis (original method)
        perform_shap_analysis(
            model_rf, X_test,
            os.path.join(args.output_dir, f"shap_summary_{args.fp_type}.png")
        )

    # Cross-validation (original method)
    print("\n" + "="*60)
    print("CROSS-VALIDATION (ORIGINAL METHODOLOGY)")
    print("="*60)

    for model_name in args.models:
        if model_name == 'knn':
            cv_scores, cv_pred = perform_cross_validation(
                train_fp, test_fp, model_type='knn', n_splits=args.cv_folds,
                fp_size=fp_size,
                n_neighbors=7, weights='distance', algorithm='brute', metric='euclidean'
            )
        elif model_name == 'rf':
            cv_scores, cv_pred = perform_cross_validation(
                train_fp, test_fp, model_type='rf', n_splits=args.cv_folds,
                fp_size=fp_size,
                n_estimators=55, random_state=args.random_state
            )

        # Save CV results
        cv_results = pd.DataFrame({
            'fold': range(1, len(cv_scores) + 1),
            'f1_score': cv_scores
        })
        cv_file = os.path.join(args.output_dir, f"cv_results_{model_name}_{args.fp_type}.csv")
        cv_results.to_csv(cv_file, index=False)
        print(f"CV results saved to: {cv_file}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
