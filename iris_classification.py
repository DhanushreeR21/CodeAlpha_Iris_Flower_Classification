

# =====================================================
# CODEALPHA TASK 1: IRIS FLOWER CLASSIFICATION
# Author: DHANUSHREE R
# =====================================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Save output files in the project folder
PROJECT_DIR = Path(__file__).resolve().parent

# STEP 2: LOAD DATASET

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

# Add flower names
df["species_name"] = df["species"].map(
    dict(enumerate(iris.target_names))
)

print("\nFIRST FIVE ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nFLOWER SPECIES:")
print(iris.target_names)

# STEP 3: DATASET INFORMATION

print("\nDATASET INFORMATION:")
df.info()

print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# STEP 4: CHECK MISSING VALUES

print("\nMISSING VALUES:")
print(df.isnull().sum())

# STEP 5: CHECK AND REMOVE DUPLICATES

print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())

df = df.drop_duplicates().copy()

print("\nDATASET SHAPE AFTER REMOVING DUPLICATES:")
print(df.shape)

# STEP 6: FLOWER SPECIES COUNT

print("\nFLOWER SPECIES COUNT:")
print(df["species_name"].value_counts())

# STEP 7: SPECIES COUNT VISUALIZATION

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="species_name",
    hue="species_name",
    legend=False
)

plt.title("Count of Iris Flower Species")
plt.xlabel("Flower Species")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(
    PROJECT_DIR / "species_count.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# STEP 8: PAIR PLOT

pair = sns.pairplot(
    df,
    vars=iris.feature_names,
    hue="species_name",
    diag_kind="hist"
)

pair.fig.suptitle(
    "Iris Flower Dataset Pair Plot",
    y=1.02
)

pair.savefig(
    PROJECT_DIR / "pairplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# STEP 9: CORRELATION HEATMAP

plt.figure(figsize=(8, 6))

sns.heatmap(
    df[iris.feature_names].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Iris Dataset Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    PROJECT_DIR / "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# STEP 10: SPLIT DATASET

X = df[iris.feature_names]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA:", X_train.shape)
print("TESTING DATA:", X_test.shape)

# STEP 11: TRAIN LOGISTIC REGRESSION MODEL

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")

# STEP 12: MAKE PREDICTIONS

y_pred = model.predict(X_test)

# STEP 13: MODEL EVALUATION

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:")
print(f"{accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        zero_division=0
    )
)

# STEP 14: CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.tight_layout()

plt.savefig(
    PROJECT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# STEP 15: INTERACTIVE FLOWER PREDICTION

print("\n====================================")
print("     IRIS FLOWER PREDICTION SYSTEM")
print("====================================")

try:
    sepal_length = float(
        input("\nEnter Sepal Length (cm): ")
    )

    sepal_width = float(
        input("Enter Sepal Width (cm): ")
    )

    petal_length = float(
        input("Enter Petal Length (cm): ")
    )

    petal_width = float(
        input("Enter Petal Width (cm): ")
    )

    # Check valid measurement ranges
    values = [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]

    if any(value <= 0 for value in values):
        raise ValueError("Measurements must be positive.")

    if not (
        4.0 <= sepal_length <= 8.5
        and 1.5 <= sepal_width <= 5.0
        and 0.5 <= petal_length <= 7.0
        and 0.0 < petal_width <= 3.0
    ):
        raise ValueError(
            "Measurements are outside the expected Iris ranges."
        )

    new_flower = pd.DataFrame(
        [values],
        columns=iris.feature_names
    )

    prediction = model.predict(new_flower)

    predicted_species = iris.target_names[prediction[0]]

    print("\n====================================")
    print("PREDICTED FLOWER SPECIES:")
    print(predicted_species.upper())
    print("====================================")

except ValueError as error:
    print("\nInvalid input:", error)

print("\nPROJECT EXECUTION COMPLETED!")
