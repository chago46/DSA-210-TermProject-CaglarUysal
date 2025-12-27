"""
Machine Learning Analysis Script

This script applies machine learning techniques to predict graduate admission outcomes:
- Random Forest / Decision Tree Classifier for approval/rejection prediction
- K-means Clustering to identify applicant groups and analyze admission patterns

The script demonstrates:
1. Data preprocessing and feature selection
2. Clustering analysis to identify applicant groups
3. Model training and evaluation
4. Visualization of results
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
import pickle
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    silhouette_score, davies_bouldin_score, mean_squared_error, r2_score
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Directories
ML_OUTPUT_DIR = Path('ml_outputs')
ML_OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load the processed dataset"""
    final_path = Path('final_combined_dataset.csv')
    merged_path = Path('merged_data_combined.csv')
    
    if final_path.exists():
        print(f"Loading data from {final_path}")
        df = pd.read_csv(final_path, low_memory=False)
    elif merged_path.exists():
        print(f"Loading data from {merged_path}")
        df = pd.read_csv(merged_path, low_memory=False)
    else:
        raise FileNotFoundError("Could not find final_combined_dataset.csv or merged_data_combined.csv")
    
    print(f"Dataset shape: {df.shape}")
    return df


def preprocess_data(df):
    """
    Preprocess data for machine learning:
    - Handle missing values
    - Select relevant features
    - Feature engineering
    - Create target variable
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    # Create a copy to avoid modifying original
    df_ml = df.copy()
    
    # Use is_accepted as target variable (1 = accepted, 0 = rejected)
    if 'is_accepted' not in df_ml.columns:
        # If is_accepted doesn't exist, try to create from 'Chance of Admit'
        if 'Chance of Admit' in df_ml.columns:
            df_ml['is_accepted'] = (df_ml['Chance of Admit'] >= 0.5).astype(int)
        else:
            raise ValueError("Neither 'is_accepted' nor 'Chance of Admit' found in dataset")
    
    # Remove rows where target is missing
    df_ml = df_ml.dropna(subset=['is_accepted'])
    print(f"Target distribution:\n{df_ml['is_accepted'].value_counts()}")
    print(f"Acceptance rate: {df_ml['is_accepted'].mean():.2%}")
    
    # Select features based on correlation analysis
    # ALL QS Score features (highest correlation with target)
    qs_score_features = [
        'qs_Academic_Reputation_Score',      # Highest correlation: 0.159
        'qs_Citations_per_Faculty_Score',     # 0.149
        'qs_Employer_Reputation_Score',      # 0.146
        'qs_Employment_Outcomes_Score',       # 0.145
        'qs_Faculty_Student_Score',           # 0.145
        'qs_Overall_Score',                   # 0.145
        'qs_International_Research_Network_Score',  # 0.102
        'qs_Sustainability_Score',            # 0.101
        'qs_International_Faculty_Score',      # 0.095
        'qs_International_Students_Score'     # 0.093
    ]
    
    # QS Rank features (inverse relationship - lower rank is better)
    qs_rank_features = [
        'qs_Academic_Reputation_Rank',
        'qs_Employer_Reputation_Rank',
        'qs_Faculty_Student_Rank',
        'qs_Citations_per_Faculty_Rank',
        'qs_International_Faculty_Rank',
        'qs_International_Students_Rank',
        'qs_Employment_Outcomes_Rank'
    ]
    
    # Academic features
    academic_features = [
        'gre_verbal', 'gre_quant', 'gre_writing', 'gre_subject',
        'ugrad_gpa', 'is_new_gre'
    ]
    
    # Categorical features to encode
    categorical_features = ['major', 'season', 'qs_Region', 'qs_SIZE', 
                           'qs_FOCUS', 'qs_RES.', 'qs_STATUS']
    
    # Start with numerical features
    numerical_features = qs_score_features + qs_rank_features + academic_features
    
    # Select features that exist in the dataset
    available_numerical = [f for f in numerical_features if f in df_ml.columns]
    
    # Create feature matrix X with numerical features
    X = df_ml[available_numerical].copy()
    
    # Encode categorical features
    print("\nEncoding categorical features...")
    label_encoders = {}
    for cat_feat in categorical_features:
        if cat_feat in df_ml.columns:
            # Fill missing values with 'Unknown'
            df_ml[cat_feat] = df_ml[cat_feat].fillna('Unknown')
            
            # Encode
            le = LabelEncoder()
            encoded_values = le.fit_transform(df_ml[cat_feat].astype(str))
            X[f'{cat_feat}_encoded'] = encoded_values
            label_encoders[cat_feat] = le
            print(f"  Encoded {cat_feat}: {len(le.classes_)} unique values")
    
    # Feature Engineering: Create composite features
    print("\nCreating engineered features...")
    
    # Combine GRE scores
    if 'gre_verbal' in X.columns and 'gre_quant' in X.columns:
        X['GRE_Total'] = X['gre_verbal'] + X['gre_quant']
        X['GRE_Average'] = (X['gre_verbal'] + X['gre_quant']) / 2
    
    # Normalize QS ranks (inverse: lower rank number = better, so we invert)
    print("Processing QS ranks...")
    for rank_col in qs_rank_features:
        if rank_col in X.columns:
            # Convert rank ranges like "101-150" to numeric (use midpoint)
            def parse_rank(val):
                if pd.isna(val):
                    return np.nan
                val_str = str(val)
                if '-' in val_str:
                    parts = val_str.split('-')
                    try:
                        return (float(parts[0]) + float(parts[1])) / 2
                    except:
                        return np.nan
                elif val_str.replace('+', '').isdigit():
                    return float(val_str.replace('+', ''))
                else:
                    return np.nan
            
            X[f'{rank_col}_numeric'] = X[rank_col].apply(parse_rank)
            # Invert: higher number = better (lower rank number = better university)
            max_rank = X[f'{rank_col}_numeric'].max()
            if not pd.isna(max_rank) and max_rank > 0:
                X[f'{rank_col}_inverted'] = max_rank - X[f'{rank_col}_numeric']
    
    # Create interaction features with top QS scores
    print("Creating interaction features...")
    
    # QS Academic Reputation (most important) interactions
    if 'qs_Academic_Reputation_Score' in X.columns:
        if 'ugrad_gpa' in X.columns:
            X['QS_Rep_GPA'] = X['qs_Academic_Reputation_Score'] * X['ugrad_gpa']
        if 'GRE_Total' in X.columns:
            X['QS_Rep_GRE'] = X['qs_Academic_Reputation_Score'] * X['GRE_Total'] / 100
    
    # QS Citations (second most important) interactions
    if 'qs_Citations_per_Faculty_Score' in X.columns and 'ugrad_gpa' in X.columns:
        X['QS_Citations_GPA'] = X['qs_Citations_per_Faculty_Score'] * X['ugrad_gpa']
    
    # Combined QS score (average of top QS scores)
    top_qs_scores = ['qs_Academic_Reputation_Score', 'qs_Citations_per_Faculty_Score',
                     'qs_Employer_Reputation_Score', 'qs_Employment_Outcomes_Score',
                     'qs_Faculty_Student_Score']
    available_top_qs = [col for col in top_qs_scores if col in X.columns]
    if len(available_top_qs) > 0:
        X['QS_Combined_Score'] = X[available_top_qs].mean(axis=1)
    
    # Handle missing values - fill with median for numerical features
    print(f"\nMissing values before imputation:")
    missing = X.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    
    # Convert all columns to numeric (coerce errors to NaN)
    print("\nConverting columns to numeric...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill missing values with median (only for numeric columns)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = X[col].median()
        if not pd.isna(median_val):
            X[col] = X[col].fillna(median_val)
        else:
            # If median is NaN, fill with 0
            X[col] = X[col].fillna(0)
    
    # Remove features with too many missing values (>50%) or constant values
    missing_ratio = X.isnull().sum() / len(X)
    X = X.loc[:, missing_ratio < 0.5]  # Keep features with <50% missing
    X = X.loc[:, X.nunique() > 1]  # Remove constant columns
    
    # Remove highly correlated features (keep one from each highly correlated pair)
    print("\nRemoving highly correlated features...")
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [column for column in upper_tri.columns 
                          if any(upper_tri[column] > 0.95)]
    if high_corr_features:
        print(f"Removing {len(high_corr_features)} highly correlated features")
        X = X.drop(high_corr_features, axis=1)
    
    print(f"\nSelected {len(X.columns)} features for ML:")
    print(X.columns.tolist())
    
    # Target variable
    y = df_ml['is_accepted']
    
    # Remove rows where all features are NaN
    valid_mask = ~X.isnull().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Target distribution after preprocessing:\n{y.value_counts()}")
    
    return X, y, df_ml[valid_mask]


def train_all_models(X, y):
    """
    Train multiple models for admission prediction:
    - Random Forest
    - Decision Tree
    - Logistic Regression
    - Linear Regression (for comparison, treating as classification)
    - Neural Network (MLPClassifier)
    Compare and select the best performing model
    """
    print("\n" + "="*60)
    print("MULTIPLE MODEL COMPARISON")
    print("Random Forest, Decision Tree, Logistic Regression, Linear Regression, Neural Network")
    print("="*60)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Standardize features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store all model results
    all_models = {}
    
    # 1. Random Forest
    print("\n[1/5] Training Random Forest Classifier...")
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [20, 30, 40],
        'min_samples_leaf': [5, 10],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced']
    }
    
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_grid_search = GridSearchCV(
        rf_base, rf_param_grid, cv=5, scoring='roc_auc',
        n_jobs=-1, verbose=0
    )
    rf_grid_search.fit(X_train, y_train)
    rf_classifier = rf_grid_search.best_estimator_
    print(f"  Best CV score: {rf_grid_search.best_score_:.4f}")
    all_models['Random Forest'] = {
        'model': rf_classifier,
        'cv_score': rf_grid_search.best_score_,
        'params': rf_grid_search.best_params_,
        'use_scaled': False
    }
    
    # 2. Decision Tree
    print("\n[2/5] Training Decision Tree Classifier...")
    dt_param_grid = {
        'max_depth': [10, 12, 15, 18],
        'min_samples_split': [30, 40, 50],
        'min_samples_leaf': [5, 10],
        'max_features': ['sqrt', None],
        'class_weight': ['balanced'],
        'criterion': ['gini', 'entropy']
    }
    
    dt_base = DecisionTreeClassifier(random_state=42)
    dt_grid_search = GridSearchCV(
        dt_base, dt_param_grid, cv=5, scoring='roc_auc',
        n_jobs=-1, verbose=0
    )
    dt_grid_search.fit(X_train, y_train)
    dt_classifier = dt_grid_search.best_estimator_
    print(f"  Best CV score: {dt_grid_search.best_score_:.4f}")
    all_models['Decision Tree'] = {
        'model': dt_classifier,
        'cv_score': dt_grid_search.best_score_,
        'params': dt_grid_search.best_params_,
        'use_scaled': False
    }
    
    # 3. Logistic Regression
    print("\n[3/5] Training Logistic Regression...")
    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced', None],
        'max_iter': [1000]
    }
    
    lr_base = LogisticRegression(random_state=42, n_jobs=-1)
    lr_grid_search = GridSearchCV(
        lr_base, lr_param_grid, cv=5, scoring='roc_auc',
        n_jobs=-1, verbose=0
    )
    lr_grid_search.fit(X_train_scaled, y_train)
    lr_classifier = lr_grid_search.best_estimator_
    print(f"  Best CV score: {lr_grid_search.best_score_:.4f}")
    all_models['Logistic Regression'] = {
        'model': lr_classifier,
        'cv_score': lr_grid_search.best_score_,
        'params': lr_grid_search.best_params_,
        'use_scaled': True
    }
    
    # 4. Linear Regression (for comparison - convert to classification)
    print("\n[4/5] Training Linear Regression (converted to classification)...")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)
    # Convert regression predictions to classification (threshold at 0.5)
    lin_reg_pred_train = (lin_reg.predict(X_train_scaled) >= 0.5).astype(int)
    lin_reg_pred_test = (lin_reg.predict(X_test_scaled) >= 0.5).astype(int)
    
    # Calculate ROC-AUC manually for Linear Regression
    from sklearn.metrics import roc_auc_score
    lin_reg_proba_train = np.clip(lin_reg.predict(X_train_scaled), 0, 1)
    lin_reg_cv_scores_list = []
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_train_scaled):
        lin_reg_fold = LinearRegression()
        lin_reg_fold.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
        y_val_proba = np.clip(lin_reg_fold.predict(X_train_scaled[val_idx]), 0, 1)
        try:
            auc = roc_auc_score(y_train.iloc[val_idx], y_val_proba)
            lin_reg_cv_scores_list.append(auc)
        except:
            lin_reg_cv_scores_list.append(0.5)  # Default if can't calculate
    
    lin_reg_cv_score = np.mean(lin_reg_cv_scores_list) if lin_reg_cv_scores_list else 0.5
    print(f"  Best CV score: {lin_reg_cv_score:.4f}")
    all_models['Linear Regression'] = {
        'model': lin_reg,
        'cv_score': lin_reg_cv_score,
        'params': {},
        'use_scaled': True,
        'is_regression': True
    }
    
    # 5. Neural Network (MLPClassifier) - Optimized but faster
    print("\n[5/5] Training Neural Network (MLPClassifier)...")
    
    # Smaller, focused grid for faster training
    nn_param_grid = {
        'hidden_layer_sizes': [
            (100,), (150,), (200,),
            (100, 50), (150, 100), (200, 100),
            (150, 100, 50)
        ],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['adaptive'],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [1000],
        'solver': ['adam'],
        'batch_size': [64, 128]
    }
    
    nn_base = MLPClassifier(random_state=42, early_stopping=True, 
                           validation_fraction=0.1, n_iter_no_change=15,
                           max_iter=1000)
    nn_grid_search = GridSearchCV(
        nn_base, nn_param_grid, cv=3, scoring='roc_auc',  # Reduced CV folds for speed
        n_jobs=-1, verbose=0
    )
    nn_grid_search.fit(X_train_scaled, y_train)
    nn_classifier = nn_grid_search.best_estimator_
    initial_score = nn_grid_search.best_score_
    print(f"  Initial CV score: {initial_score:.4f}")
    print(f"  Best parameters: {nn_grid_search.best_params_}")
    
    # Quick refinement if promising (only if close to Random Forest)
    if initial_score > 0.64:  # Only refine if close to best
        print("  Quick refinement around best parameters...")
        best_hidden = nn_grid_search.best_params_['hidden_layer_sizes']
        best_activation = nn_grid_search.best_params_['activation']
        best_alpha = nn_grid_search.best_params_['alpha']
        best_lr = nn_grid_search.best_params_['learning_rate_init']
        
        # Small refined grid
        if isinstance(best_hidden, tuple) and len(best_hidden) == 1:
            hidden_variants = [
                best_hidden,
                (int(best_hidden[0]*1.2),),
                (int(best_hidden[0]*0.8),),
                (best_hidden[0], best_hidden[0]//2)
            ]
        else:
            hidden_variants = [best_hidden, (150, 100), (200, 100)]
        
        nn_param_grid_refined = {
            'hidden_layer_sizes': hidden_variants[:4],
            'activation': [best_activation],
            'alpha': [best_alpha * 0.5, best_alpha, best_alpha * 1.5],
            'learning_rate': ['adaptive'],
            'learning_rate_init': [best_lr * 0.5, best_lr, best_lr * 2],
            'max_iter': [1500],
            'solver': ['adam'],
            'batch_size': [64, 128]
        }
        
        nn_refined = MLPClassifier(random_state=42, early_stopping=True,
                                  validation_fraction=0.1, n_iter_no_change=15,
                                  max_iter=1500)
        nn_refined_search = GridSearchCV(
            nn_refined, nn_param_grid_refined, cv=3, scoring='roc_auc',
            n_jobs=-1, verbose=0
        )
        nn_refined_search.fit(X_train_scaled, y_train)
        
        if nn_refined_search.best_score_ > initial_score:
            print(f"  Refined CV score: {nn_refined_search.best_score_:.4f} (improved!)")
            nn_classifier = nn_refined_search.best_estimator_
            nn_grid_search.best_score_ = nn_refined_search.best_score_
            nn_grid_search.best_params_ = nn_refined_search.best_params_
        else:
            print(f"  Refined score: {nn_refined_search.best_score_:.4f} (no improvement)")
    
    all_models['Neural Network'] = {
        'model': nn_classifier,
        'cv_score': nn_grid_search.best_score_,
        'params': nn_grid_search.best_params_,
        'use_scaled': True
    }
    
    # Find best model
    best_model_name = max(all_models.keys(), key=lambda k: all_models[k]['cv_score'])
    best_model_info = all_models[best_model_name]
    classifier = best_model_info['model']
    model_name = best_model_name
    
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    for name, info in sorted(all_models.items(), key=lambda x: x[1]['cv_score'], reverse=True):
        marker = " [BEST]" if name == best_model_name else ""
        print(f"{name:25s}: CV Score = {info['cv_score']:.4f}{marker}")
    print(f"{'='*60}")
    print(f"\n[+] Best Model: {best_model_name} (CV Score: {best_model_info['cv_score']:.4f})")
    
    # Try Ensemble Methods (Voting and Stacking) to potentially improve accuracy
    print("\n" + "="*60)
    print("ENSEMBLE METHODS")
    print("="*60)
    
    # Prepare models for ensemble (use top 3 models)
    top_models = sorted(all_models.items(), key=lambda x: x[1]['cv_score'], reverse=True)[:3]
    print(f"\nCreating ensemble from top 3 models: {[m[0] for m in top_models]}")
    
    ensemble_models = []
    for name, info in top_models:
        if info['use_scaled']:
            # For scaled models, we'll use them with scaled data
            ensemble_models.append((name.lower().replace(' ', '_'), info['model']))
        else:
            ensemble_models.append((name.lower().replace(' ', '_'), info['model']))
    
    # Voting Classifier (soft voting for probability)
    print("\n[1/2] Training Voting Classifier (Soft Voting)...")
    voting_classifier = VotingClassifier(
        estimators=ensemble_models,
        voting='soft',
        n_jobs=-1
    )
    
    # Use scaled data if any model needs it
    use_scaled_ensemble = any(all_models[m[0]]['use_scaled'] for m in top_models)
    X_train_ensemble = X_train_scaled if use_scaled_ensemble else X_train
    X_test_ensemble = X_test_scaled if use_scaled_ensemble else X_test
    
    voting_classifier.fit(X_train_ensemble, y_train)
    voting_cv_scores = cross_val_score(voting_classifier, X_train_ensemble, y_train, 
                                       cv=3, scoring='roc_auc', n_jobs=-1)
    voting_cv_score = voting_cv_scores.mean()
    print(f"  Voting Classifier CV score: {voting_cv_score:.4f}")
    
    # Stacking Classifier
    print("\n[2/2] Training Stacking Classifier...")
    # Use Logistic Regression as meta-learner
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)
    stacking_classifier = StackingClassifier(
        estimators=ensemble_models,
        final_estimator=meta_learner,
        cv=3,
        n_jobs=-1
    )
    stacking_classifier.fit(X_train_ensemble, y_train)
    stacking_cv_scores = cross_val_score(stacking_classifier, X_train_ensemble, y_train,
                                        cv=3, scoring='roc_auc', n_jobs=-1)
    stacking_cv_score = stacking_cv_scores.mean()
    print(f"  Stacking Classifier CV score: {stacking_cv_score:.4f}")
    
    # Add ensemble models to comparison
    all_models['Voting Classifier'] = {
        'model': voting_classifier,
        'cv_score': voting_cv_score,
        'params': {'voting': 'soft', 'models': [m[0] for m in top_models]},
        'use_scaled': use_scaled_ensemble
    }
    
    all_models['Stacking Classifier'] = {
        'model': stacking_classifier,
        'cv_score': stacking_cv_score,
        'params': {'meta_learner': 'LogisticRegression', 'models': [m[0] for m in top_models]},
        'use_scaled': use_scaled_ensemble
    }
    
    # Re-evaluate best model including ensembles
    best_model_name = max(all_models.keys(), key=lambda k: all_models[k]['cv_score'])
    best_model_info = all_models[best_model_name]
    classifier = best_model_info['model']
    model_name = best_model_name
    
    print(f"\n{'='*60}")
    print(f"UPDATED MODEL COMPARISON (Including Ensembles)")
    print(f"{'='*60}")
    for name, info in sorted(all_models.items(), key=lambda x: x[1]['cv_score'], reverse=True):
        marker = " [BEST]" if name == best_model_name else ""
        print(f"{name:25s}: CV Score = {info['cv_score']:.4f}{marker}")
    print(f"{'='*60}")
    print(f"\n[+] Final Best Model: {best_model_name} (CV Score: {best_model_info['cv_score']:.4f})")
    
    # Use scaled or unscaled data based on model
    if best_model_info['use_scaled']:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    else:
        X_train_final = X_train
        X_test_final = X_test
    
    # Cross-validation scores
    if best_model_info.get('is_regression', False):
        cv_scores = cross_val_score(classifier, X_train_final, y_train, cv=5, scoring='r2')
        print(f"\nCross-validation R2 scores: {cv_scores}")
        print(f"Mean CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    else:
        cv_scores = cross_val_score(classifier, X_train_final, y_train, cv=5, scoring='accuracy')
        print(f"\nCross-validation accuracy scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Predictions
    if best_model_info.get('is_regression', False):
        y_train_pred_proba = classifier.predict(X_train_final)
        y_test_pred_proba = classifier.predict(X_test_final)
        y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
        y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
        y_test_proba = np.clip(y_test_pred_proba, 0, 1)  # Clip to [0, 1] for ROC
    else:
        y_train_pred = classifier.predict(X_train_final)
        y_test_pred = classifier.predict(X_test_final)
        y_test_proba = classifier.predict_proba(X_test_final)[:, 1]
    
    # Evaluate model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\n{model_name} Performance:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    
    # Classification report
    class_report = classification_report(y_test, y_test_pred, 
                                        target_names=['Rejected', 'Accepted'],
                                        output_dict=True)
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                              target_names=['Rejected', 'Accepted']))
    
    # Feature importance (handle ensembles that might not have feature_importances_)
    try:
        if hasattr(classifier, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': classifier.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(classifier, 'coef_'):
            # For linear models, use absolute coefficients
            coef = classifier.coef_[0] if classifier.coef_.ndim > 1 else classifier.coef_
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(coef)
            }).sort_values('importance', ascending=False)
        else:
            # For ensembles, try to get from base models
            if hasattr(classifier, 'estimators_'):
                # Average importance from all estimators
                importances = []
                for est in classifier.estimators_:
                    if hasattr(est, 'feature_importances_'):
                        importances.append(est.feature_importances_)
                if importances:
                    avg_importance = np.mean(importances, axis=0)
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': avg_importance
                    }).sort_values('importance', ascending=False)
                else:
                    # Fallback: equal importance
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': np.ones(len(X.columns)) / len(X.columns)
                    }).sort_values('importance', ascending=False)
            else:
                # Fallback: equal importance
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.ones(len(X.columns)) / len(X.columns)
                }).sort_values('importance', ascending=False)
    except Exception as e:
        print(f"Warning: Could not extract feature importance: {e}")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.ones(len(X.columns)) / len(X.columns)
        }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Feature Importance
    top_features = feature_importance.head(15)
    axes[0, 0].barh(range(len(top_features)), top_features['importance'])
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features['feature'])
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_title(f'Top 15 Feature Importances ({model_name})')
    axes[0, 0].invert_yaxis()
    
    # 2. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['Rejected', 'Accepted'],
                yticklabels=['Rejected', 'Accepted'])
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')
    axes[0, 1].set_title(f'Confusion Matrix ({model_name})')
    
    # 3. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    axes[0, 2].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {test_roc_auc:.3f})')
    axes[0, 2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0, 2].set_xlabel('False Positive Rate')
    axes[0, 2].set_ylabel('True Positive Rate')
    axes[0, 2].set_title('ROC Curve')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # 4. Prediction Distribution
    try:
        if best_model_info.get('is_regression', False):
            prediction_proba = np.clip(classifier.predict(X_test_final), 0, 1)
        else:
            prediction_proba = classifier.predict_proba(X_test_final)[:, 1]
    except:
        prediction_proba = y_test_proba
    axes[1, 0].hist(prediction_proba[y_test == 0], bins=30, alpha=0.5, 
                    label='Rejected', color='red')
    axes[1, 0].hist(prediction_proba[y_test == 1], bins=30, alpha=0.5, 
                    label='Accepted', color='green')
    axes[1, 0].set_xlabel('Predicted Probability of Acceptance')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Probability Distribution')
    axes[1, 0].legend()
    axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    
    # 5. Feature Importance Distribution
    axes[1, 1].bar(range(len(feature_importance)), 
                   feature_importance['importance'].sort_values(ascending=False))
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Importance')
    axes[1, 1].set_title('Feature Importance Distribution')
    
    # 6. Metrics Comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    values = [test_accuracy, test_precision, test_recall, test_f1, test_roc_auc]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    bars = axes[1, 2].bar(metrics, values, color=colors, alpha=0.7)
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Model Performance Metrics')
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, values):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(ML_OUTPUT_DIR / 'decision_tree_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nModel visualizations saved to {ML_OUTPUT_DIR / 'decision_tree_analysis.png'}")
    plt.close()
    
    # Create model comparison visualizations
    print("\nCreating model comparison visualizations...")
    create_model_comparison_visualizations(all_models, X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test, scaler)
    
    # Save the best model
    print("\nSaving best model...")
    model_save_path = ML_OUTPUT_DIR / 'best_model.pkl'
    scaler_save_path = ML_OUTPUT_DIR / 'scaler.pkl'
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(classifier, f)
    
    # Save scaler if model uses scaled data
    if best_model_info['use_scaled']:
        with open(scaler_save_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"  Model saved to: {model_save_path}")
        print(f"  Scaler saved to: {scaler_save_path}")
    else:
        print(f"  Model saved to: {model_save_path}")
    
    # Save model metadata
    model_metadata = {
        'model_name': model_name,
        'cv_score': float(best_model_info['cv_score']),
        'test_accuracy': float(test_accuracy),
        'test_roc_auc': float(test_roc_auc),
        'best_params': str(best_model_info['params']),
        'use_scaled': best_model_info['use_scaled'],
        'feature_names': list(X.columns)
    }
    
    with open(ML_OUTPUT_DIR / 'model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    print(f"  Model metadata saved to: {ML_OUTPUT_DIR / 'model_metadata.json'}")
    
    return classifier, feature_importance, {
        'model_name': model_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'best_params': best_model_info['params'],
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'all_models': all_models
    }


def create_model_comparison_visualizations(all_models, X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test, scaler):
    """Create comprehensive comparison visualizations for all models"""
    print("  Creating ROC curves comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. ROC Curves for all models
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_models)))
    
    for (name, info), color in zip(sorted(all_models.items(), key=lambda x: x[1]['cv_score'], reverse=True), colors):
        model = info['model']
        use_scaled = info['use_scaled']
        X_test_final = X_test_scaled if use_scaled else X_test
        
        try:
            if info.get('is_regression', False):
                y_proba = np.clip(model.predict(X_test_final), 0, 1)
            else:
                y_proba = model.predict_proba(X_test_final)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax1.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2, color=color)
        except Exception as e:
            print(f"    Warning: Could not plot ROC for {name}: {e}")
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)')
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curves - All Models Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. CV Scores Comparison
    ax2 = axes[0, 1]
    model_names = [name for name, _ in sorted(all_models.items(), key=lambda x: x[1]['cv_score'], reverse=True)]
    cv_scores = [all_models[name]['cv_score'] for name in model_names]
    colors_bar = ['green' if score == max(cv_scores) else 'steelblue' for score in cv_scores]
    
    bars = ax2.barh(range(len(model_names)), cv_scores, color=colors_bar, alpha=0.8, edgecolor='black')
    ax2.set_yticks(range(len(model_names)))
    ax2.set_yticklabels(model_names, fontsize=9)
    ax2.set_xlabel('Cross-Validation ROC-AUC Score', fontsize=11)
    ax2.set_title('Model Performance Comparison (CV Scores)', fontsize=12, fontweight='bold')
    ax2.set_xlim([min(cv_scores) - 0.05, max(cv_scores) + 0.05])
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, score) in enumerate(zip(bars, cv_scores)):
        ax2.text(score + 0.002, i, f'{score:.4f}', va='center', fontsize=8, fontweight='bold')
    
    # 3. Test Accuracy Comparison
    ax3 = axes[1, 0]
    test_accuracies = []
    for name in model_names:
        model = all_models[name]['model']
        use_scaled = all_models[name]['use_scaled']
        X_test_final = X_test_scaled if use_scaled else X_test
        
        try:
            if all_models[name].get('is_regression', False):
                y_pred = (model.predict(X_test_final) >= 0.5).astype(int)
            else:
                y_pred = model.predict(X_test_final)
            acc = accuracy_score(y_test, y_pred)
            test_accuracies.append(acc)
        except:
            test_accuracies.append(0.0)
    
    colors_bar_acc = ['green' if acc == max(test_accuracies) else 'coral' for acc in test_accuracies]
    bars = ax3.barh(range(len(model_names)), test_accuracies, color=colors_bar_acc, alpha=0.8, edgecolor='black')
    ax3.set_yticks(range(len(model_names)))
    ax3.set_yticklabels(model_names, fontsize=9)
    ax3.set_xlabel('Test Accuracy', fontsize=11)
    ax3.set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
    ax3.set_xlim([min(test_accuracies) - 0.05, max(test_accuracies) + 0.05])
    ax3.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, acc) in enumerate(zip(bars, test_accuracies)):
        ax3.text(acc + 0.005, i, f'{acc:.4f}', va='center', fontsize=8, fontweight='bold')
    
    # 4. Model Complexity vs Performance
    ax4 = axes[1, 1]
    # Estimate complexity (number of parameters or model type complexity)
    complexities = []
    for name in model_names:
        model = all_models[name]['model']
        if hasattr(model, 'n_estimators'):  # Random Forest
            complexities.append(model.n_estimators * 10)
        elif hasattr(model, 'hidden_layer_sizes'):  # Neural Network
            total_neurons = sum(model.hidden_layer_sizes) if isinstance(model.hidden_layer_sizes, tuple) else model.hidden_layer_sizes[0]
            complexities.append(total_neurons)
        elif 'Ensemble' in name or 'Voting' in name or 'Stacking' in name:
            complexities.append(50)  # Medium complexity
        else:
            complexities.append(10)  # Simple models
    
    scatter = ax4.scatter(complexities, cv_scores, s=200, alpha=0.6, c=colors_bar, edgecolors='black', linewidth=2)
    for i, name in enumerate(model_names):
        ax4.annotate(name, (complexities[i], cv_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax4.set_xlabel('Model Complexity (Estimated)', fontsize=11)
    ax4.set_ylabel('CV ROC-AUC Score', fontsize=11)
    ax4.set_title('Model Complexity vs Performance', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ML_OUTPUT_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Model comparison saved to {ML_OUTPUT_DIR / 'model_comparison.png'}")
    plt.close()


def perform_clustering_analysis(X, y, X_scaled=None):
    """
    Perform K-means clustering to identify applicant groups
    and analyze their admission patterns
    """
    print("\n" + "="*60)
    print("K-MEANS CLUSTERING ANALYSIS")
    print("="*60)
    
    # Standardize features for clustering
    if X_scaled is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    # Select key features for clustering (most important academic and QS features)
    key_features = [
        'qs_Academic_Reputation_Score', 'qs_Citations_per_Faculty_Score',
        'gre_verbal', 'gre_quant', 'ugrad_gpa', 'qs_Overall_Score'
    ]
    available_key_features = [f for f in key_features if f in X.columns]
    
    if len(available_key_features) > 0:
        # Use key features for clustering
        feature_indices = [list(X.columns).index(f) for f in available_key_features]
        X_cluster = X_scaled[:, feature_indices]
        feature_names = available_key_features
    else:
        # Use all features if key features not available
        X_cluster = X_scaled
        feature_names = list(X.columns)
    
    print(f"Using {len(feature_names)} features for clustering")
    
    # Determine optimal number of clusters
    print("\nDetermining optimal number of clusters...")
    silhouette_scores = []
    davies_bouldin_scores = []
    inertias = []
    K_range = range(2, 8)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_cluster)
        silhouette_scores.append(silhouette_score(X_cluster, cluster_labels))
        davies_bouldin_scores.append(davies_bouldin_score(X_cluster, cluster_labels))
        inertias.append(kmeans.inertia_)
    
    # Find optimal k (highest silhouette score, lowest DB score)
    optimal_k_idx = np.argmax(silhouette_scores)
    optimal_k = K_range[optimal_k_idx]
    print(f"Optimal number of clusters (by silhouette score): {optimal_k}")
    print(f"Silhouette score: {silhouette_scores[optimal_k_idx]:.4f}")
    print(f"Davies-Bouldin score: {davies_bouldin_scores[optimal_k_idx]:.4f}")
    
    # Apply K-means with optimal k
    print(f"\nApplying K-means with {optimal_k} clusters...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)
    
    # Calculate final metrics
    silhouette_avg = silhouette_score(X_cluster, cluster_labels)
    db_score = davies_bouldin_score(X_cluster, cluster_labels)
    print(f"Final Silhouette Score: {silhouette_avg:.4f}")
    print(f"Final Davies-Bouldin Score: {db_score:.4f}")
    
    # Analyze clusters
    df_clustered = pd.DataFrame(X_scaled, columns=X.columns)
    df_clustered['cluster'] = cluster_labels
    df_clustered['is_accepted'] = y.values
    
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS")
    print("="*60)
    cluster_stats = []
    
    for cluster_id in range(optimal_k):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        acceptance_rate = cluster_data['is_accepted'].mean()
        cluster_size = len(cluster_data)
        
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {cluster_size} ({cluster_size/len(df_clustered)*100:.1f}%)")
        print(f"  Acceptance Rate: {acceptance_rate:.2%}")
        
        # Calculate mean values of key features for this cluster
        if len(available_key_features) > 0:
            print(f"  Average feature values:")
            for feat in available_key_features:
                if feat in cluster_data.columns:
                    mean_val = cluster_data[feat].mean()
                    print(f"    {feat}: {mean_val:.3f}")
        
        cluster_stats.append({
            'cluster': cluster_id,
            'size': cluster_size,
            'acceptance_rate': acceptance_rate
        })
    
    # Visualizations
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Elbow Method
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax1.set_ylabel('Inertia', fontsize=11)
    ax1.set_title('Elbow Method for Optimal k', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Silhouette Scores
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(K_range, silhouette_scores, marker='o', color='green', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax2.set_ylabel('Silhouette Score', fontsize=11)
    ax2.set_title('Silhouette Score vs Number of Clusters', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Davies-Bouldin Scores
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(K_range, davies_bouldin_scores, marker='o', color='purple', linewidth=2, markersize=8)
    ax3.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
    ax3.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax3.set_ylabel('Davies-Bouldin Score', fontsize=11)
    ax3.set_title('Davies-Bouldin Score vs Number of Clusters', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cluster Acceptance Rates
    ax4 = fig.add_subplot(gs[1, 0])
    cluster_acceptance = df_clustered.groupby('cluster')['is_accepted'].agg(['mean', 'count'])
    colors_cluster = plt.cm.Set3(np.linspace(0, 1, optimal_k))
    bars = ax4.bar(range(optimal_k), cluster_acceptance['mean'], color=colors_cluster, alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Cluster', fontsize=11)
    ax4.set_ylabel('Acceptance Rate', fontsize=11)
    ax4.set_title('Acceptance Rate by Cluster', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(optimal_k))
    ax4.set_ylim([0, 1])
    for i, (rate, count) in enumerate(zip(cluster_acceptance['mean'], cluster_acceptance['count'])):
        ax4.text(i, rate + 0.02, f'{rate:.1%}\n(n={count})', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Cluster Sizes
    ax5 = fig.add_subplot(gs[1, 1])
    cluster_sizes = df_clustered['cluster'].value_counts().sort_index()
    bars = ax5.bar(range(optimal_k), cluster_sizes.values, color=colors_cluster, alpha=0.8, edgecolor='black')
    ax5.set_xlabel('Cluster', fontsize=11)
    ax5.set_ylabel('Number of Applicants', fontsize=11)
    ax5.set_title('Cluster Sizes', fontsize=12, fontweight='bold')
    ax5.set_xticks(range(optimal_k))
    for i, size in enumerate(cluster_sizes.values):
        ax5.text(i, size + max(cluster_sizes.values)*0.01, f'{size}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Cluster Distribution (Pie Chart)
    ax6 = fig.add_subplot(gs[1, 2])
    sizes = cluster_sizes.values
    labels = [f'Cluster {i}\n({sizes[i]} applicants)' for i in range(optimal_k)]
    ax6.pie(sizes, labels=labels, colors=colors_cluster, autopct='%1.1f%%', 
           startangle=90, textprops={'fontsize': 9})
    ax6.set_title('Cluster Distribution', fontsize=12, fontweight='bold')
    
    # 7. 2D PCA Visualization
    ax7 = fig.add_subplot(gs[2, :])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster)
    
    scatter = ax7.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                         cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    ax7.set_xlabel(f'First Principal Component (explains {pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
    ax7.set_ylabel(f'Second Principal Component (explains {pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
    ax7.set_title('K-means Clustering (2D PCA Projection)', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax7)
    cbar.set_label('Cluster', fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ML_OUTPUT_DIR / 'clustering_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nClustering visualizations saved to {ML_OUTPUT_DIR / 'clustering_analysis.png'}")
    plt.close()
    
    # Create detailed cluster comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    cluster_comparison = pd.DataFrame(cluster_stats)
    
    x = np.arange(optimal_k)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cluster_comparison['size'], width, 
                   label='Cluster Size', color='steelblue', alpha=0.8)
    ax2_twin = ax.twinx()
    bars2 = ax2_twin.bar(x + width/2, cluster_comparison['acceptance_rate']*100, width,
                       label='Acceptance Rate (%)', color='coral', alpha=0.8)
    
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Applicants', fontsize=12, fontweight='bold', color='steelblue')
    ax2_twin.set_ylabel('Acceptance Rate (%)', fontsize=12, fontweight='bold', color='coral')
    ax.set_title('Cluster Size vs Acceptance Rate', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Cluster {i}' for i in range(optimal_k)])
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2_twin.tick_params(axis='y', labelcolor='coral')
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + max(cluster_comparison['size'])*0.01,
               f'{int(bar1.get_height())}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2_twin.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 1,
                     f'{bar2.get_height():.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(ML_OUTPUT_DIR / 'cluster_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Cluster comparison saved to {ML_OUTPUT_DIR / 'cluster_comparison.png'}")
    plt.close()
    
    return kmeans, scaler, cluster_labels, {
        'n_clusters': optimal_k,
        'silhouette_score': silhouette_avg,
        'davies_bouldin_score': db_score,
        'cluster_acceptance_rates': cluster_acceptance['mean'].to_dict(),
        'cluster_sizes': cluster_sizes.to_dict(),
        'cluster_stats': cluster_stats
    }


def generate_report(dt_results, feature_importance, clustering_results=None):
    """Generate a text report of ML analysis results"""
    report_path = ML_OUTPUT_DIR / 'ml_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MACHINE LEARNING ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"{dt_results['model_name'].upper()} CLASSIFIER RESULTS\n")
        f.write("-"*60 + "\n")
        f.write(f"Model Type: {dt_results['model_name']}\n")
        f.write(f"Best Hyperparameters: {dt_results['best_params']}\n\n")
        
        # Model comparison
        if 'all_models' in dt_results:
            f.write("MODEL COMPARISON (CV Scores):\n")
            f.write("-"*60 + "\n")
            for name, info in sorted(dt_results['all_models'].items(), 
                                    key=lambda x: x[1]['cv_score'], reverse=True):
                marker = " [SELECTED]" if name == dt_results['model_name'] else ""
                f.write(f"{name:25s}: {info['cv_score']:.4f}{marker}\n")
            f.write("\n")
        f.write(f"Cross-Validation Accuracy: {dt_results['cv_mean']:.4f} (+/- {dt_results['cv_std']:.4f})\n")
        f.write(f"Training Accuracy: {dt_results['train_accuracy']:.4f}\n")
        f.write(f"Test Accuracy: {dt_results['test_accuracy']:.4f}\n")
        f.write(f"Test Precision: {dt_results['test_precision']:.4f}\n")
        f.write(f"Test Recall: {dt_results['test_recall']:.4f}\n")
        f.write(f"Test F1-Score: {dt_results['test_f1']:.4f}\n")
        f.write(f"Test ROC-AUC: {dt_results['test_roc_auc']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(dt_results['confusion_matrix']) + "\n")
        f.write("\nDetailed Classification Report (Test Set):\n")
        f.write("-"*60 + "\n")
        if 'classification_report' in dt_results:
            report = dt_results['classification_report']
            f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n")
            f.write("-"*60 + "\n")
            for class_name in ['Rejected', 'Accepted']:
                if class_name in report:
                    metrics = report[class_name]
                    f.write(f"{class_name:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                           f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<12}\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Accuracy':<15} {'':<12} {'':<12} {report['accuracy']:<12.4f} "
                   f"{int(report['macro avg']['support']):<12}\n")
            f.write(f"{'Macro Avg':<15} {report['macro avg']['precision']:<12.4f} "
                   f"{report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f} "
                   f"{int(report['macro avg']['support']):<12}\n")
            f.write(f"{'Weighted Avg':<15} {report['weighted avg']['precision']:<12.4f} "
                   f"{report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f} "
                   f"{int(report['weighted avg']['support']):<12}\n")
        f.write("\n")
        
        f.write("TOP 20 MOST IMPORTANT FEATURES\n")
        f.write("-"*60 + "\n")
        top_20 = feature_importance.head(20)
        for idx, row in top_20.iterrows():
            f.write(f"{row['feature']:40s}: {row['importance']:.6f}\n")
        
        if clustering_results:
            f.write("\n\n" + "="*60 + "\n")
            f.write("K-MEANS CLUSTERING RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Number of Clusters: {clustering_results['n_clusters']}\n")
            f.write(f"Silhouette Score: {clustering_results['silhouette_score']:.4f}\n")
            f.write(f"Davies-Bouldin Score: {clustering_results['davies_bouldin_score']:.4f}\n")
            f.write("\nAcceptance Rates by Cluster:\n")
            for cluster, rate in clustering_results['cluster_acceptance_rates'].items():
                size = clustering_results['cluster_sizes'].get(cluster, 0)
                f.write(f"  Cluster {cluster}: {rate:.2%} (n={size})\n")
    
    print(f"\nML analysis report saved to {report_path}")


def main():
    """Main function to run ML analysis"""
    print("="*60)
    print("MACHINE LEARNING ANALYSIS")
    print("Graduate Admission Prediction using Decision Tree")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Preprocess data
    X, y, df_processed = preprocess_data(df)
    
    # Perform Clustering Analysis
    clustering_model, clustering_scaler, cluster_labels, clustering_results = perform_clustering_analysis(X, y)
    
    # Train all models and compare
    dt_model, feature_importance, dt_results = train_all_models(X, y)
    
    # Generate report
    generate_report(dt_results, feature_importance, clustering_results)
    
    print("\n" + "="*60)
    print("ML ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to {ML_OUTPUT_DIR}/")
    print("  - decision_tree_analysis.png")
    print("  - model_comparison.png")
    print("  - clustering_analysis.png")
    print("  - cluster_comparison.png")
    print("  - ml_analysis_report.txt")
    print("  - best_model.pkl (saved model for future predictions)")
    print("  - model_metadata.json (model information)")


if __name__ == "__main__":
    main()

