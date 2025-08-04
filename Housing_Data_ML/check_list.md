# Machine Learning Project Workflow Checklist

## 🔹 1. Problem Definition
- [ ] Define the objective (regression or classification)
- [ ] Identify target variable and relevant features

## 🔹 2. Data Acquisition
- [ ] Load data (CSV, database, API, etc.)

## 🔹 3. Initial Data Inspection
- [ ] View .head(), .info(), and .describe()
- [ ] Check data types and value ranges

## 🔹 4. Data Cleaning
- [ ] Handle missing values
- [ ] Remove or fix erroneous entries
- [ ] Convert data types if necessary

## 🔹 5. Exploratory Data Analysis (EDA)
- [ ] Univariate plots (histograms, countplots)
- [ ] Bivariate plots (scatterplots, boxplots)
- [ ] Identify outliers and skewness
- [ ] Check feature correlations

## 🔹 6. Feature Engineering
- [ ] Create new features from existing ones
- [ ] Combine or transform columns
- [ ] Normalize or scale if needed

## 🔹 7. Categorical Encoding
- [ ] One-hot encode or ordinal encode features
- [ ] Handle rare categories

## 🔹 8. Train-Test Split
- [ ] Split into train and test sets

## 🔹 9. Pipeline Setup
- [ ] Use Pipeline and ColumnTransformer
- [ ] Include scaling and encoding steps

## 🔹 10. Model Training
- [ ] Train baseline models (e.g. LinearRegression, RandomForest)
- [ ] Use cross-validation

## 🔹 11. Model Evaluation
- [ ] Evaluate with RMSE, MAE, R² (for regression)
- [ ] Visualize residuals and predictions

## 🔹 12. Model Tuning
- [ ] Try hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- [ ] Experiment with different algorithms

## 🔹 13. Final Evaluation
- [ ] Evaluate final model on test set
- [ ] Save metrics and plots

## 🔹 14. Deployment (optional)
- [ ] Export model with joblib or pickle
- [ ] Build API (Flask/FastAPI) or app (Streamlit)
- [ ] Write final documentation
