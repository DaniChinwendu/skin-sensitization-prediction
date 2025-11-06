# skin-sensitization-prediction
Python toolkit for molecular feature extraction, similarity mapping, and machine learning–based prediction of skin sensitization. Includes RDKit-based analysis, fingerprint visualization, Random Forest and KNN modeling, and SHAP interpretability for mechanistic insight and regulatory toxicology.
# Skin Sensitization Prediction: KNN vs Random Forest

Companion code for: *"Optimizing Skin Sensitization Prediction: A Comparative Analysis of KNN vs Random Forest"*

**Authors:** Daniel C. Ukaegbu1, Karolina Kopanska1, Peter Ranslow2, Alexandra Maertens1*   
**Affiliation:** 1 Center for Alternatives to Animal Testing (CAAT), Johns Hopkins Bloomberg School of Public Health, Baltimore, Maryland 21205, United States
**Affiliation:** 2 Consortium for Environmental Risk Management (CERM), Hallowell, Maine 04347, United States

## Overview

This repository contains code for analyzing molecular fingerprints and predicting skin sensitization using machine learning models.

## Features

- **Molecular_analysis**: Detect and analyze conjugated and aromatic systems in molecules
- **Rdkit_FP_Visualization**: Visualize Morgan and AtomPair fingerprint bits
- **Similarity_Analysis**: Compute Tanimoto similarity matrices with multiple fingerprint types (Morgan, MACCS, AtomPair, PubChem)
- **Train_and_Test**: Train and evaluate KNN and Random Forest classifiers with cross-validation and SHAP analysis

  🧪 Molecular Fingerprint Analysis Toolkit

This repository contains scripts for training and testing machine learning models, performing molecular similarity analysis, and visualizing chemical fingerprints using RDKit.

🚀 How to Use

You can run all scripts directly in Google Colab using any modern web browser (Chrome, Firefox, Edge, etc.).

1️⃣ Train and Test

Open the notebook and upload your training and testing datasets.

Update the following parameters as needed:

random_state – to control reproducibility

fingerprint_type – choose your preferred molecular fingerprint (e.g., Morgan, Avalon, RDKit)

n_estimators – number of trees for Random Forest

n_neighbors – number of neighbors for KNN

Run the notebook to train and evaluate your models.

2️⃣ Similarity Analysis

Upload the required files when prompted.

Update the following parameters as appropriate:

threshold – similarity threshold value

fp_type – fingerprint type (e.g., Morgan, AtomPair, Avalon)

n_bits – number of bits for the fingerprint vector

For Morgan fingerprints, adjust additional parameters such as radius.

The script calculates molecular similarity scores and visualizes key relationships.

3️⃣ RDKit Fingerprint Visualization

Use this notebook to visualize active fingerprint bits for specific molecules.

You can modify the input SMILES strings to explore different structures.

Supports visualization for Morgan and Atom Pair fingerprints.

4️⃣ Molecular Analysis

Designed for analyzing conjugated and aromatic systems.

Upload your data files (df_train, df_test, etc.) when prompted.

Run the notebook to explore structural and physicochemical characteristics of molecules.

⚙️ Requirements

Python ≥ 3.9

RDKit

pandas

scikit-learn

matplotlib / seaborn
