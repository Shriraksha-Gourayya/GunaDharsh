# ==============================================================
# NUTRISCAN AI
# Mango Condition Classification Using Spectral Data
#
# Input:
#   18 AS7265X spectral channels
#
# Target:
#   RIPE / ROTTEN
#
# Models:
#   1. Random Forest
#   2. Support Vector Machine (SVM)
#
# Preprocessing:
#   - Data validation
#   - Missing/invalid value handling
#   - Grouped train-test split by mango_id
#   - Median imputation when required
#   - Standardization for SVM
#   - Class weighting for imbalance
#
# Outputs:
#   models/mango_model.joblib
#   models/model_comparison.csv
# ==============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ==============================================================
# 1. CONFIGURATION
# ==============================================================

DATASET_PATH = "mango_dataset.csv"
MODEL_DIRECTORY = "models"

RANDOM_STATE = 42
TEST_SIZE = 0.25

os.makedirs(MODEL_DIRECTORY, exist_ok=True)


# ==============================================================
# 2. DEFINE SPECTRAL FEATURES
# ==============================================================

SPECTRAL_FEATURES = [
    "nm410",
    "nm435",
    "nm460",
    "nm485",
    "nm510",
    "nm535",
    "nm560",
    "nm585",
    "nm610",
    "nm645",
    "nm680",
    "nm705",
    "nm730",
    "nm760",
    "nm810",
    "nm860",
    "nm900",
    "nm940",
]

TARGET_COLUMN = "label"
GROUP_COLUMN = "mango_id"

EXPECTED_LABELS = {"RIPE", "ROTTEN"}


# ==============================================================
# 3. LOAD DATASET
# ==============================================================

print("\n" + "=" * 70)
print("NUTRISCAN AI - MANGO CONDITION CLASSIFICATION")
print("=" * 70)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset not found: {DATASET_PATH}\n"
        "Make sure mango_dataset.csv is in the project folder."
    )

df = pd.read_csv(DATASET_PATH)

print("\nSTEP 1: DATASET INFORMATION")
print("-" * 70)

print("Dataset shape:", df.shape)
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

required_columns = (
    SPECTRAL_FEATURES
    + [TARGET_COLUMN, GROUP_COLUMN]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from dataset: {missing_columns}"
    )

print("\nOriginal label distribution:")
print(df[TARGET_COLUMN].value_counts(dropna=False))

print("\nMissing values in original dataset:")
print(df.isnull().sum().sum())

print("\nCompletely duplicated rows:")
print(df.duplicated().sum())


# ==============================================================
# 4. VALIDATE LABELS AND MANGO IDS
# ==============================================================

print("\nSTEP 2: VALIDATING LABELS AND MANGO IDS")
print("-" * 70)

original_row_count = len(df)

# Normalize label formatting without changing its meaning.
df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype("string")
    .str.strip()
    .str.upper()
)

# Remove rows without a usable target label or group ID.
# These rows cannot be used for supervised grouped training.
df = df.dropna(
    subset=[TARGET_COLUMN, GROUP_COLUMN]
).copy()

df[GROUP_COLUMN] = (
    df[GROUP_COLUMN]
    .astype(str)
    .str.strip()
)

# Reject unexpected class labels instead of silently changing them.
observed_labels = set(df[TARGET_COLUMN].unique())

unexpected_labels = observed_labels - EXPECTED_LABELS

if unexpected_labels:
    raise ValueError(
        "Unexpected labels found: "
        f"{unexpected_labels}\n"
        "Expected only RIPE and ROTTEN."
    )

if df[GROUP_COLUMN].eq("").any():
    raise ValueError(
        "Empty mango IDs detected. Please correct them first."
    )

print("Rows before validation:", original_row_count)
print("Rows after removing unusable labels/IDs:", len(df))

print("\nValidated label distribution:")
print(df[TARGET_COLUMN].value_counts())

print("\nUnique mango IDs:", df[GROUP_COLUMN].nunique())

print("\nRows per mango ID:")
print(df.groupby(GROUP_COLUMN).size().to_string())


# ==============================================================
# 5. PREPROCESS SPECTRAL MEASUREMENTS
# ==============================================================

print("\nSTEP 3: PREPROCESSING SPECTRAL FEATURES")
print("-" * 70)

# Convert spectral measurements to numeric.
# Unexpected text becomes NaN and is reported below.
for column in SPECTRAL_FEATURES:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Replace infinite values with NaN.
# The model pipeline can then impute missing feature values.
df[SPECTRAL_FEATURES] = df[SPECTRAL_FEATURES].replace(
    [np.inf, -np.inf],
    np.nan
)

X = df[SPECTRAL_FEATURES].copy()

print(
    "Missing spectral values:",
    int(X.isnull().sum().sum())
)

print(
    "Rows containing at least one missing spectral value:",
    int(X.isnull().any(axis=1).sum())
)

print(
    "Infinite spectral values:",
    int(np.isinf(X.to_numpy()).sum())
)

# Negative values are reported, not automatically deleted.
negative_count = int((X < 0).sum().sum())

print("Negative spectral values:", negative_count)

if negative_count > 0:
    print(
        "WARNING: Negative values were found. "
        "Investigate them before accepting the model results."
    )

# Print statistics before any imputation or scaling.
print("\nSpectral feature summary:")
print(X.describe().T.to_string())


# ==============================================================
# 6. ENCODE TARGET LABELS
# ==============================================================

print("\nSTEP 4: PREPARING FEATURES AND TARGET")
print("-" * 70)

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df[TARGET_COLUMN]
)

groups = df[GROUP_COLUMN].to_numpy()

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

print("\nLabel encoding:")

for index, label in enumerate(label_encoder.classes_):
    print(f"{label} = {index}")

print("\nInput features:")
print(SPECTRAL_FEATURES)


# ==============================================================
# 7. GROUPED TRAIN-TEST SPLIT
# ==============================================================

print("\nSTEP 5: SPLITTING DATA")
print("-" * 70)

# All rows belonging to a given mango ID remain together.
# This prevents the same ID from appearing in both partitions.
#
# IMPORTANT:
# This is valid only if each mango_id represents a distinct
# physical mango, as assumed for this experiment.

unique_groups = np.unique(groups)

if len(unique_groups) < 4:
    raise ValueError(
        "Too few distinct mango IDs for this initial evaluation."
    )

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y[train_indices]
y_test = y[test_indices]

train_groups = set(groups[train_indices])
test_groups = set(groups[test_indices])

shared_groups = train_groups.intersection(test_groups)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("Training mango IDs:", len(train_groups))
print("Testing mango IDs:", len(test_groups))

print("Shared mango IDs:", len(shared_groups))

if shared_groups:
    raise RuntimeError(
        "Data leakage detected: mango IDs occur in both sets."
    )

print("\nTraining label distribution:")
print(
    pd.Series(
        label_encoder.inverse_transform(y_train)
    ).value_counts()
)

print("\nTesting label distribution:")
print(
    pd.Series(
        label_encoder.inverse_transform(y_test)
    ).value_counts()
)

if len(np.unique(y_train)) < 2:
    raise ValueError(
        "The training split contains only one class. "
        "A different grouped split is required."
    )

if len(np.unique(y_test)) < 2:
    raise ValueError(
        "The test split contains only one class. "
        "Choose a grouped split that includes both classes."
    )


# ==============================================================
# 8. DEFINE RANDOM FOREST AND SVM
# ==============================================================

print("\nSTEP 6: DEFINING MACHINE-LEARNING MODELS")
print("-" * 70)

# Random Forest:
# - Median imputation for missing measurements.
# - Class weighting for class imbalance.
# - No standardization required.

random_forest = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

# SVM:
# - Median imputation.
# - Standardization fitted on training data only.
# - Class weighting for class imbalance.

svm = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

models = {
    "Random Forest": random_forest,
    "SVM": svm,
}


# ==============================================================
# 9. TRAIN AND EVALUATE MODELS
# ==============================================================

print("\nSTEP 7: TRAINING AND EVALUATING MODELS")
print("-" * 70)

results = []
trained_models = {}

class_indices = np.arange(
    len(label_encoder.classes_)
)

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("TRAINING:", model_name)
    print("=" * 70)

    # Train using training data only.
    model.fit(X_train, y_train)

    # Evaluate using held-out mango IDs.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    print(f"\nAccuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Macro precision:   {macro_precision:.4f}")
    print(f"Macro recall:      {macro_recall:.4f}")
    print(f"Macro F1-score:    {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=class_indices,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=class_indices,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual {label}"
            for label in label_encoder.classes_
        ],
        columns=[
            f"Predicted {label}"
            for label in label_encoder.classes_
        ],
    )

    print("Confusion matrix:")
    print(matrix_df.to_string())

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_accuracy,
            "Macro Precision": macro_precision,
            "Macro Recall": macro_recall,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
        }
    )

    trained_models[model_name] = model


# ==============================================================
# 10. COMPARE MODELS
# ==============================================================

print("\n" + "=" * 70)
print("STEP 8: MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
).reset_index(drop=True)

print(results_df.to_string(index=False))

# Select the model with the highest macro F1-score on this
# preliminary held-out evaluation.
#
# A stronger final evaluation should use grouped validation
# on training data and reserve the test set for one final check.

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nSelected model:", best_model_name)


# ==============================================================
# 11. SAVE MODEL AND METADATA
# ==============================================================

print("\nSTEP 9: SAVING MODEL")
print("-" * 70)

model_bundle = {
    "model": best_model,
    "model_name": best_model_name,
    "feature_names": SPECTRAL_FEATURES,
    "label_encoder": label_encoder,
    "class_names": list(label_encoder.classes_),
    "target_column": TARGET_COLUMN,
    "group_column": GROUP_COLUMN,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "class_weight": "balanced",
}

model_path = os.path.join(
    MODEL_DIRECTORY,
    "mango_model.joblib",
)

comparison_path = os.path.join(
    MODEL_DIRECTORY,
    "model_comparison.csv",
)

joblib.dump(
    model_bundle,
    model_path
)

results_df.to_csv(
    comparison_path,
    index=False
)

print("Saved model:", model_path)
print("Saved comparison:", comparison_path)


# ==============================================================
# 12. FINAL SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("Dataset rows used:", len(df))
print("Spectral features:", len(SPECTRAL_FEATURES))
print("Distinct mango IDs:", df[GROUP_COLUMN].nunique())
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Selected model:", best_model_name)

print("\nFinal comparison:")
print(results_df.to_string(index=False))

print(
    "\nIMPORTANT NOTES:"
    "\n1. Class weighting was used instead of duplicating samples."
    "\n2. The original spectral measurements were not rescaled "
    "in the CSV."
    "\n3. SVM scaling was fitted on training data only."
    "\n4. Evaluation was grouped by mango_id."
    "\n5. Results measure agreement with recorded labels."
    "\n6. Internal rot detection requires independently verified "
    "fruit-condition labels."
)

print("\nNutriScan AI training script finished successfully.")