# Customer Churn Prediction

## Overview

Customer churn refers to customers discontinuing the use of a company's service. Predicting churn is important for businesses because acquiring new customers is often more expensive than retaining existing ones.

This project focuses on predicting customer churn using machine learning techniques. By analyzing customer behavior data such as service usage, payment information, and account details, the model identifies customers who are likely to leave the service.

The system helps businesses take proactive actions to improve customer retention and reduce revenue loss.

---

## Objectives

* Analyze customer data to understand patterns related to churn.
* Apply machine learning algorithms to predict whether a customer will churn or stay.
* Compare different classification models and select the best-performing model.
* Use XGBoost to improve prediction accuracy.
* Provide visual insights and predictions through a simple Streamlit interface.

---

## Dataset

The dataset used in this project contains customer information including:

* Customer demographics
* Service subscription details
* Billing and payment information
* Account details
* Customer churn status

The dataset is provided in **CSV format** and is used for training and evaluating the machine learning models.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Streamlit

---

## Project Workflow

1. Data Collection (CSV dataset)
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Encoding (One-Hot Encoding)
5. Model Training using Machine Learning Algorithms
6. Model Evaluation
7. Customer Churn Prediction
8. Visualization using Streamlit Dashboard

---

## Machine Learning Algorithms Used

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

Among these models, **XGBoost achieved the best performance** and was selected as the final model for churn prediction.

---

## Project Structure

```
Customer_Churn_Prediction
│
├── dataset
│   └── customer_churn.csv
│
├── notebooks
│   └── churn_analysis.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/SivaKothapalli4234/Customer_Churn_Prediction.git
```

Navigate to the project directory:

```
cd Customer_Churn_Prediction
```

Install required libraries:

```
pip install -r requirements.txt
```

---

## Running the Application

Run the Streamlit application using the following command:

```
streamlit run app.py
```

This will open the web application where users can input customer information and predict whether the customer is likely to churn.

---

## Results

The trained machine learning model predicts whether a customer will churn or stay with the company. The prediction results help businesses identify potential churn customers and take preventive actions to improve customer retention.

---

## Future Work

* Use larger and more diverse datasets to improve prediction accuracy.
* Apply advanced machine learning or deep learning techniques.
* Integrate real-time customer data for continuous churn monitoring.
* Enhance the Streamlit dashboard with more interactive visualizations.
* Deploy the system as a cloud-based or web application.

---

## Conclusion

This project demonstrates how machine learning techniques can be used to analyze customer behavior and predict churn effectively. The developed system helps organizations identify customers who may leave and supports better business decision-making for customer retention strategies.

---

## Author

Siva Kothapalli
