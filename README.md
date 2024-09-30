## Statistical-Modeling
 Developing models for datasets

# Market Share Analysis: Predicting Customer Switching Behavior
This project involves analyzing energy market share data to identify trends and model customer behavior, specifically focusing on predicting customer switching patterns using various machine learning models.

# Table of Contents
Data Preparation
Trend and Correlation Analysis
Predictive Modeling
Evaluation and Improvements
How to Run the Code
Technologies Used
Data Preparation
We start by loading the market share data from an Excel file into a pandas DataFrame. The dataset includes monthly records of customer numbers and energy usage, broken down by affiliated and non-affiliated customers.

Company: Name of the company (e.g., Oncor)
Month: Month of the data entry
Year: Year of the data entry
NOC_Affiliate: Number of customers affiliated with the company
MWHS_Affiliate: Megawatt hours used by affiliates
Avg_Res_kWH_Usage: Average residential kilowatt-hour usage, etc.
Trend and Correlation Analysis
To understand the data better, we first performed a descriptive analysis and visualizations. For instance, we examined the distribution of customers by company and studied the relationship between the Affiliate Rate (percentage of affiliated customers) and Average Residential kWh Usage.

We calculated the correlation between key variables like Affiliate Rate and Avg_Res_kWH_Usage, and visualized this using a correlation matrix heatmap.

#Predictive Modeling
We created features like Switching_Behavior, which flags instances where there was a decrease in the number of affiliated customers compared to the previous month. This column is used as the target for modeling.

We then built the following models to predict customer switching behavior:

#Linear Regression

Predicted customer switching behavior based on the Affiliate Rate and Avg_Res_kWH_Usage.
Results indicated a low R-squared score, suggesting the need for more complex models.
Logistic Regression

Achieved better performance compared to linear regression, with accuracy evaluated using the test set.
Random Forest Classifier

Performed well, demonstrating higher accuracy than logistic regression.
Further tuning was done using grid search and cross-validation.

# Evaluation and Improvements
We used the Random Forest Classifier as a final model and improved its performance using SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance. We also used Polynomial Features to capture non-linear relationships in the data. Finally, we performed hyperparameter tuning using GridSearchCV, which identified the best parameters for our random forest model.

# Prerequisites
Make sure you have the following packages installed:

pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
You can install these dependencies using pip:

bash
Copy code
pip install pandas matplotlib seaborn scikit-learn imbalanced-learn
Steps
Load the data into the pandas DataFrame:

Adjust the file path in the pd.read_excel() function to point to your local Excel file.
Run the notebook or Python script.

Visualize the trends and run the machine learning models as described above.

# Technologies Used
Python for data analysis and modeling
pandas for data manipulation
seaborn and matplotlib for data visualization
scikit-learn for machine learning
imbalanced-learn for handling class imbalance
Future Work
Explore additional features to improve model performance.
Implement advanced machine learning models like Gradient Boosting or XGBoost.
Build a more comprehensive customer switching prediction system.
