# Breast Cancer Prognosis

## 1. Objective  
The primary objective of this study is to evaluate and compare various machine learning models to determine which is best suited for predicting whether a patient is affected by breast cancer. The study aims to identify the most effective model for this use case, rather than simply building a single classifier.  

Key components of this analysis include:  
- **Data Pre-processing**: Ensuring the dataset is clean, normalized, and prepared for modeling.  
- **Data Visualization**: Exploring the dataset through visual methods to better understand feature distributions, correlations, and class separations.  
- **Model Evaluation**: Assessing and comparing the performance of different machine learning algorithms.  

The findings will offer insights into the suitability of various models for this critical medical application, emphasizing accuracy, robustness, and interpretability.  

---

## 2. Dataset Description  
- **Dataset**: [Breast Cancer Wisconsin (Prognostic)](https://archive.ics.uci.edu/dataset/16/breast+cancer+wisconsin+prognostic).  
Dataset consists of 35 columns, where first row is a unique patient ID, second row is an outcome (a label) and the rest are features. There are 198 samples of examined patients for training / testing the models.
More information can be found in the [notebook](jupyter_notebook/data_notebook.ipynb).

---

## 3. Methodology and Solution   
This section outlines the approach and methodology adopted to analyze and compare machine learning models for breast cancer prediction. The study involves systematic data preprocessing, the creation of multiple datasets, and training with various models to identify the most effective solution.  

### 3.1 Data Preprocessing  
To ensure the quality and usability of the dataset, the following preprocessing steps were performed:  
1. **Normalization**: All numerical features were normalized to a standard range to ensure consistency across features.  
2. **Categorical Encoding**: The categorical attribute (target variable) was encoded into binary values (0 and 1).  
3. **Null Value Handling**: Any missing or null values were addressed by imputation or elimination to avoid inconsistencies in the dataset.  
4. **Feature Selection**: The top 10 most informative features were selected using feature selection techniques, ensuring that they have a significant impact on classification performance.  

### 3.2 Creation of Additional Datasets  
As part of the methodology, three distinct datasets were created for model training:  
1. **Original Dataset**: The unaltered dataset after basic preprocessing.  
2. **Normalized Dataset**: The dataset where all features have been normalized.  
3. **Reduced Dataset**: A dataset consisting of only the 10 most significant features identified during feature selection.  

These datasets enable a comparative analysis of model performance under varying data representations.  

### 3.3 Selected Models and Training Process  
The study evaluated the performance of six machine learning models on the three datasets:  
1. **Decision Tree**  
2. **Random Forest**  
3. **Extra Tree**  
4. **Support Vector Classifier (SVC)**  
5. **Nu-Support Vector Classifier (NuSVC)**  
6. **Linear Support Vector Classifier (LinearSVC)**  

**Training and Evaluation**:  
- Each model was trained on all three datasets, allowing for a robust comparison across different data preprocessing strategies.  
- Hyperparameter tuning was performed for each model to identify the best configuration and optimize performance.  

### 3.4 Outcome  
The training and evaluation process aims to:  
- Compare the performance of each model across all datasets.  
- Identify which model and dataset combination yields the best results.  
- Gain insights into the impact of data preprocessing and feature selection on classification accuracy and overall model effectiveness.

---

## 4. Data Preprocessing  
Data preprocessing is a crucial step to ensure that the dataset is clean, consistent, and ready for effective model training. The following steps were performed:  

### 4.1 Handling Missing and Null Values  
- Any missing or null values in the dataset were addressed.  
- Depending on the nature of the missing data, values were either imputed (using statistical methods) or eliminated to maintain dataset integrity.  

### 4.2 Normalization  
- All numerical features were normalized to a consistent scale, ensuring that large variations in feature values do not bias model training.  
- This step is particularly important for algorithms sensitive to feature magnitude, such as Support Vector Classifiers.  

### 4.3 Categorical Encoding  
- The target variable, which indicates whether a patient is affected by breast cancer, was encoded into binary values:  
  - 0: Not affected (benign).  
  - 1: Affected (malignant).  

### 4.4 Feature Selection  
- To reduce dimensionality and focus on the most relevant information, the top 10 most significant features were selected from the original dataset.  
- Feature selection was performed using statistical or algorithmic techniques, ensuring that these features have the highest predictive power for the classification task.  

### 4.5 Creation of Multiple Datasets  
Following preprocessing, three distinct datasets were created to evaluate the impact of preprocessing on model performance:  
1. **Original Dataset**: Retains all features after basic preprocessing.  
2. **Normalized Dataset**: All features normalized to a consistent scale.  
3. **Reduced Dataset**: A subset containing only the top 10 selected features.  

This preprocessing strategy enables a comprehensive analysis of how different data representations influence model performance, contributing to the identification of the best predictive model.  

---

## 5. Evaluation Metrics  
To assess and compare the performance of the machine learning models, several evaluation metrics were considered. Among them, **F1-score** was selected as the most significant metric for this study due to the following reasons:  

### 5.1 Importance of F1-score  
- **F1-score** provides a harmonic mean of precision and recall, making it particularly effective in scenarios where the class distribution is imbalanced.  
- It ensures that both false positives and false negatives are taken into account, offering a balanced view of model performance.  
- Given the medical nature of this problem, minimizing false negatives (i.e., undiagnosed cancer cases) is critical, and F1-score helps highlight models that perform well in this regard.  

### 5.2 Additional Metrics  
While F1-score is the primary focus, the following metrics were also evaluated to provide a comprehensive performance overview:  
- **Accuracy**: The ratio of correctly predicted instances to the total number of instances.  
- **Precision**: The proportion of true positive predictions among all positive predictions.  
- **Recall (Sensitivity)**: The proportion of true positives identified out of all actual positive cases.  
- **Area Under the Curve (AUC)**: Evaluates the ability of the model to distinguish between classes at various threshold settings, especially useful for binary classification problems.

---

## 6. Experimental Results  

---

## 7. Conclusion   

---

## 8. Citations
- **Dataset**:
  ```
  @misc{breast_cancer_wisconsin_(prognostic)_16,
    author       = {Wolberg, William, Street, W., and Mangasarian, Olvi},
    title        = {{Breast Cancer Wisconsin (Prognostic)}},
    year         = {1995},
    howpublished = {UCI Machine Learning Repository},
    note         = {{DOI}: https://doi.org/10.24432/C5GK50}
  }
  ```
