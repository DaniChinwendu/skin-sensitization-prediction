# skin-sensitization-prediction
Python toolkit for molecular feature extraction, similarity mapping, and machine learning–based prediction of skin sensitization. Includes RDKit-based analysis, fingerprint visualization, Random Forest and KNN modeling, and SHAP interpretability for mechanistic insight and regulatory toxicology.
# Skin Sensitization Prediction: KNN vs Random Forest

Companion code for: *"Optimizing Skin Sensitization Prediction: A Comparative Analysis of KNN vs Random Forest"*

**Authors:** Daniel C. Ukaegbu, ..., Thomas Hartung, Alexandra Maertens  
**Affiliation:** Center for Alternatives to Animal Testing (CAAT), Johns Hopkins Bloomberg School of Public Health

## Overview

This repository contains code for analyzing molecular fingerprints and predicting skin sensitization using machine learning models.

## Features

- **Conjugation Analysis**: Detect and analyze conjugated systems in molecules
- **Fingerprint Visualization**: Visualize Morgan and AtomPair fingerprint bits
- **Similarity Analysis**: Compute Tanimoto similarity matrices with multiple fingerprint types (Morgan, MACCS, AtomPair, PubChem)
- **Model Training**: Train and evaluate KNN and Random Forest classifiers with cross-validation and SHAP analysis

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/skin-sensitization-prediction.git
cd skin-sensitization-prediction

# Install dependencies
pip install -r requirements.txt
