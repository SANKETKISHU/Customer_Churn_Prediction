# Customer Churn Prediction

## Project Overview

This project aims to predict customer churn for a subscription-based service, such as a telecommunications company, using machine learning techniques. By identifying customers at risk of churn, the company can take proactive measures to retain them, such as offering incentives or personalized promotions.

## Dataset

The dataset `Customer_Churn_Prediction_data.csv` includes customer demographics, account information, and service usage details.

## Steps Involved

1. **Data Loading and Preprocessing:**
   - Loaded the dataset.
   - Handled missing values.
   - Encoded categorical variables.
   - Scaled numerical features.

2. **Feature Selection and Target Variable:**
   - Defined features (X) and target (y).

3. **Data Splitting:**
   - Split data into training and testing sets (70% training, 30% testing).

4. **Model Training:**
   - Trained a Random Forest Classifier.

5. **Model Evaluation:**
   - Generated predictions.
   - Evaluated model using accuracy, precision, recall, F1-score, ROC-AUC score, and confusion matrix.
   - Plotted ROC Curve.

## Results

- **Accuracy:** 80%
- **ROC-AUC Score:** 0.70

The model provides reasonable predictions, helping to identify at-risk customers.

## Usage

To run the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Customer-Churn-Prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Customer-Churn-Prediction
    ```
3. Install the required packages:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
4. Run the script:
    ```bash
    python Customer_Churn_Prediction.py
    ```

## Conclusion

The project successfully predicts customer churn, allowing the company to take proactive measures to retain customers.
