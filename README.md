# Diabetes Prediction Model Project



## Overview
This project implements a machine learning model to predict diabetes using the Pima Indian Diabetes dataset. The project includes comprehensive exploratory data analysis (EDA), data preprocessing, and model training using K-Nearest Neighbors (KNN) algorithm.

## Dataset
The project uses the `diabetes.csv` dataset containing health metrics for diabetes prediction. The dataset includes the following features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (target variable: 0 = No Diabetes, 1 = Diabetes)

## Project Structure
```
diabetes prediction project/
├── Diabetes_Prediction.ipynb    # Main Jupyter notebook
├── diabetes.csv                 # Dataset file
└── README.md                   # Project documentation
```

## Features

### 1. Exploratory Data Analysis (EDA)
- **Data Overview**: Shape, data types, and basic statistics
- **Missing Values Check**: Verification of data completeness
- **Duplicate Detection**: Identification and handling of duplicate records
- **Statistical Summary**: Descriptive statistics for all features

### 2. Data Visualization
- **Target Distribution**: Count plot showing diabetes vs non-diabetes cases
- **Outlier Detection**: Box plots for identifying outliers in each feature
- **Feature Relationships**: Pair plots colored by outcome
- **Distribution Analysis**: Histograms with KDE for each feature
- **Correlation Analysis**: Heatmap showing feature correlations

### 3. Data Preprocessing
- **Outlier Removal**: Capping method using IQR (Interquartile Range)
- **Feature Scaling**: StandardScaler for normalizing feature values
- **Train-Test Split**: 80-20 split for model training and evaluation

### 4. Model Training
- **Algorithm**: K-Nearest Neighbors (KNN) Classifier
- **Hyperparameter Tuning**: Testing K values from 1 to 14
- **Optimization**: Finding optimal K value based on accuracy scores

### 5. Model Evaluation
- **Performance Metrics**: Accuracy score, confusion matrix, classification report
- **Visualization**: K-value vs accuracy plot for both training and test sets

## Requirements

### Python Libraries
```python
pandas
numpy
matplotlib
seaborn
scikit-learn
warnings
```

## Installation

1. Clone or download the project files
2. Install required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Diabetes_Prediction.ipynb
   ```

2. **Run the cells sequentially** to:
   - Load and explore the dataset
   - Perform data visualization
   - Preprocess the data
   - Train the KNN model
   - Evaluate model performance

## Key Results

- **Data Shape**: 768 samples with 9 features
- **Data Quality**: No missing values, some duplicates removed
- **Outlier Treatment**: Applied capping method using IQR
- **Best K Value**: K=18 (based on test accuracy)
- **Model Performance**: Evaluated using confusion matrix and classification report

## Methodology

1. **Data Loading**: Import dataset using pandas
2. **EDA**: Comprehensive analysis of data structure and patterns
3. **Visualization**: Multiple plot types for data understanding
4. **Preprocessing**: 
   - Outlier removal using capping
   - Feature standardization
5. **Model Training**: KNN with hyperparameter optimization
6. **Evaluation**: Multiple metrics for performance assessment

## Model Performance Visualization

The project includes visualization of:
- Training vs testing accuracy across different K values
- Confusion matrix for final model evaluation
- Feature correlation heatmap
- Distribution plots for all features

## Future Enhancements

Potential improvements for the project:
- Try other machine learning algorithms (Random Forest, SVM, etc.)
- Implement cross-validation for more robust evaluation
- Feature selection techniques
- Ensemble methods
- Web application deployment

## Contributing

Feel free to fork this project and submit pull requests for any improvements.

## License

This project is open source and available under the MIT License.

---

**Note**: This project is for educational purposes and should not be used for actual medical diagnosis. Always consult healthcare professionals for medical advice.
