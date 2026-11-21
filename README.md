
# Iris Flower Classification 🌸

## CodeAlpha Data Science Internship – Task 1

### Project Overview
This project uses Machine Learning to classify Iris flowers into three species:
- Setosa
- Versicolor
- Virginica

The classification is performed using the Logistic Regression algorithm based on flower measurements.

### Dataset Information
The Iris dataset is obtained from Scikit-learn.

- Total records: 150
- Records after duplicate removal: 149
- Input features: 4
- Target classes: 3

### Features
1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

### Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Project Workflow
1. Import required libraries.
2. Load the Iris dataset.
3. Explore the dataset.
4. Check missing values and duplicates.
5. Remove duplicate records.
6. Perform Exploratory Data Analysis (EDA).
7. Create data visualizations.
8. Split the dataset into training and testing sets.
9. Train a Logistic Regression model.
10. Evaluate model performance.
11. Predict flower species using user input.

### Visualizations

The project generates four graphs:

- species_count.png – Flower species distribution.
- pairplot.png – Relationships between flower measurements.
- correlation_heatmap.png – Correlation between features.
- confusion_matrix.png – Model evaluation results.

### Model Performance

Algorithm: Logistic Regression

Training records: 119

Testing records: 30

Accuracy: 96.67%

### Sample Prediction

Input:
- Sepal Length: 5.1 cm
- Sepal Width: 3.4 cm
- Petal Length: 1.4 cm
- Petal Width: 0.2 cm

Predicted Species: SETOSA

### How to Run the Project

1. Clone or download this repository.
2. Install the required libraries:

   pip install -r requirements.txt

3. Run the Python program:

   python iris_classification.py

4. Enter the four flower measurements when prompted.
5. View the predicted flower species.

### Project Author
DHANUSHREE R

CodeAlpha Data Science Internship – Task 1



