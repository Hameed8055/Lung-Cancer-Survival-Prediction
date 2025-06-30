# Lung-Cancer-Survival-Prediction
Certainly! Here's the **entire README** in one block for your project:

````markdown
# Lung Cancer Survival Prediction

## **Objective**
The goal of this project is to predict whether a lung cancer patient will survive or not based on various clinical features such as age, gender, smoking status, BMI, family history, treatment type, and others. The dataset is highly imbalanced, with a significant number of **"survived"** cases compared to **"not survived"** cases, which makes the prediction task more challenging.

## **Technologies Used**
- **Python 3.7+**
- **Libraries**:
  - **pandas** for data manipulation and analysis
  - **numpy** for numerical computing
  - **matplotlib** and **seaborn** for data visualization
  - **scikit-learn** for machine learning algorithms and model evaluation
  - **xgboost** for gradient boosting model
  - **lightgbm** for gradient boosting model
  - **catboost** for categorical feature handling in gradient boosting
  - **imbalanced-learn** for handling class imbalance with SMOTE (Synthetic Minority Over-sampling Technique)

## **Steps to Run the Project**

### 1. **Clone the Repository**
Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/lung-cancer-survival-prediction.git
cd lung-cancer-survival-prediction
````

### 2. **Install Dependencies**

Ensure you have the necessary libraries installed. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

### 3. **Load the Dataset**

The dataset (CSV or other formats) is required for this project. Ensure the dataset is in the project directory or update the file path in the code where the dataset is loaded.

Example:

```python
data = pd.read_csv('lung_cancer_data.csv')
```

### 4. **Run the Model**

Run the script to train and evaluate the model. Make sure that the preprocessing steps (such as handling missing values, scaling, and encoding) are performed before training the model.

For training and evaluation, you can use the following Python script:

```bash
python lung_cancer_model.py
```

## **Model Overview**

### **1. Data Preprocessing**

* **Missing Value Handling**: Missing numerical values were imputed with the **median** and categorical values with the **mode**.
* **Feature Engineering**: The **`age_group`** column was converted into numerical values (e.g., `0-30` → 30).
* **Scaling**: Numerical features such as age, BMI, and cholesterol level were scaled using **StandardScaler**.

### **2. Handling Class Imbalance**

* **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to generate synthetic data for the minority class, thereby balancing the dataset.

### **3. Models Used**

Several machine learning models were tested to predict lung cancer survival:

* **Random Forest Classifier**: An ensemble model using multiple decision trees.
* **Logistic Regression**: A baseline model to compare performance.
* **XGBoost**: A gradient boosting model known for handling class imbalance effectively.
* **LightGBM**: Another gradient boosting model that focuses on fast computation.
* **CatBoost**: A gradient boosting model optimized for categorical features.

### **4. Model Evaluation**

The model's performance was evaluated using **cross-validation** and metrics such as:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **ROC-AUC**

### **5. Hyperparameter Tuning**

* Hyperparameter tuning was done using **RandomizedSearchCV** for **XGBoost**, **LightGBM**, and **Logistic Regression**.
* **Threshold adjustment** was applied to improve recall for the minority class.

## **Results**

The models were evaluated using the following metrics:

* **XGBoost**:

  * **Accuracy**: \~78%
  * **ROC-AUC**: \~0.50 (indicating random guessing)
  * **Precision for Class 1** (Not Survived): 0.00
  * **Recall for Class 1** (Not Survived): 0.00

* **LightGBM**:

  * Similar results to **XGBoost**, with low precision and high recall for the minority class.

## **Next Steps**

1. **Model Improvement**:

   * Increase **`scale_pos_weight`** further to focus on the minority class.
   * Experiment with **ADASYN** or **Tomek Links** to improve class balancing.
   * **Tune hyperparameters** using **GridSearchCV** or **RandomizedSearchCV** for further optimization.

2. **Try Alternative Models**:

   * **CatBoost** might perform better for categorical data.
   * **Deep learning** models can be explored for better performance.

3. **Model Deployment**:

   * Save the trained model using **pickle** or **joblib**.
   * Create an API using **Flask** or **FastAPI** for real-time predictions.

---

## **Folder Structure**

```
lung-cancer-survival-prediction/
├── data/                      # Raw and cleaned dataset
│   └── lung_cancer_data.csv
├── models/                    # Trained models
│   └── xgboost_model.pkl
│   └── lightgbm_model.pkl
├── src/                       # Python scripts
│   └── lung_cancer_model.py   # Model training and evaluation script
│   └── preprocessing.py       # Data preprocessing functions
├── requirements.txt           # Dependencies for the project
└── README.md                  # Project documentation
```

---

### **Conclusion**

The project successfully tested various models for predicting lung cancer survival with a highly imbalanced dataset. **XGBoost** and **LightGBM** showed the most promise but still faced challenges in classifying the minority class effectively. Further tuning and adjustments are needed to optimize performance, especially for the **minority class**.
