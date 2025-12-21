"""
Machine Learning Analysis Script

This script applies machine learning techniques to predict graduate admission outcomes:
- Decision Tree Classifier for approval/rejection prediction

The script demonstrates:
1. Data preprocessing and feature selection
2. Decision Tree model training and evaluation
3. Visualization of results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
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


def train_decision_tree(X, y):
    """
    Train a Decision Tree classifier for admission prediction with hyperparameter tuning
    Also try Random Forest for comparison
    """
    print("\n" + "="*60)
    print("DECISION TREE & RANDOM FOREST CLASSIFIERS")
    print("="*60)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Try Random Forest first (usually performs better)
    print("\nTraining Random Forest Classifier...")
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
    
    print(f"Random Forest - Best parameters: {rf_grid_search.best_params_}")
    print(f"Random Forest - Best CV score: {rf_grid_search.best_score_:.4f}")
    
    rf_classifier = rf_grid_search.best_estimator_
    
    # Also train Decision Tree for comparison
    print("\nTraining Decision Tree Classifier...")
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
    
    print(f"Decision Tree - Best parameters: {dt_grid_search.best_params_}")
    print(f"Decision Tree - Best CV score: {dt_grid_search.best_score_:.4f}")
    
    dt_classifier = dt_grid_search.best_estimator_
    
    # Choose the better model
    if rf_grid_search.best_score_ >= dt_grid_search.best_score_:
        print("\n✓ Random Forest performs better - using it as final model")
        classifier = rf_classifier
        model_name = "Random Forest"
    else:
        print("\n✓ Decision Tree performs better - using it as final model")
        classifier = dt_classifier
        model_name = "Decision Tree"
    
    # Cross-validation scores
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\nCross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Predictions
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    y_test_proba = classifier.predict_proba(X_test)[:, 1]
    
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
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                              target_names=['Rejected', 'Accepted']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': classifier.feature_importances_
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
    prediction_proba = dt_classifier.predict_proba(X_test)[:, 1]
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
    
    return classifier, feature_importance, {
        'model_name': model_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc,
        'confusion_matrix': cm,
        'best_params': rf_grid_search.best_params_ if model_name == "Random Forest" else dt_grid_search.best_params_,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'rf_cv_score': rf_grid_search.best_score_,
        'dt_cv_score': dt_grid_search.best_score_
    }


def generate_report(dt_results, feature_importance):
    """Generate a text report of ML analysis results"""
    report_path = ML_OUTPUT_DIR / 'ml_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MACHINE LEARNING ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"{dt_results['model_name'].upper()} CLASSIFIER RESULTS\n")
        f.write("-"*60 + "\n")
        f.write(f"Model Type: {dt_results['model_name']}\n")
        f.write(f"Best Hyperparameters: {dt_results['best_params']}\n")
        f.write(f"Random Forest CV Score: {dt_results.get('rf_cv_score', 'N/A'):.4f}\n")
        f.write(f"Decision Tree CV Score: {dt_results.get('dt_cv_score', 'N/A'):.4f}\n\n")
        f.write(f"Cross-Validation Accuracy: {dt_results['cv_mean']:.4f} (+/- {dt_results['cv_std']:.4f})\n")
        f.write(f"Training Accuracy: {dt_results['train_accuracy']:.4f}\n")
        f.write(f"Test Accuracy: {dt_results['test_accuracy']:.4f}\n")
        f.write(f"Test Precision: {dt_results['test_precision']:.4f}\n")
        f.write(f"Test Recall: {dt_results['test_recall']:.4f}\n")
        f.write(f"Test F1-Score: {dt_results['test_f1']:.4f}\n")
        f.write(f"Test ROC-AUC: {dt_results['test_roc_auc']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(dt_results['confusion_matrix']) + "\n\n")
        
        f.write("TOP 20 MOST IMPORTANT FEATURES\n")
        f.write("-"*60 + "\n")
        top_20 = feature_importance.head(20)
        for idx, row in top_20.iterrows():
            f.write(f"{row['feature']:40s}: {row['importance']:.6f}\n")
    
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
    
    # Train Decision Tree
    dt_model, feature_importance, dt_results = train_decision_tree(X, y)
    
    # Generate report
    generate_report(dt_results, feature_importance)
    
    print("\n" + "="*60)
    print("ML ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to {ML_OUTPUT_DIR}/")
    print("  - decision_tree_analysis.png")
    print("  - ml_analysis_report.txt")


if __name__ == "__main__":
    main()

