# Import libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#taking dataset 
from typing import cast
from sklearn.datasets import fetch_california_housing
from sklearn.utils import Bunch
#split data into train and test sets
from sklearn.model_selection import train_test_split
#import standard scaling
from sklearn.preprocessing import StandardScaler
#importing linear regression model
from sklearn.linear_model import LinearRegression
#import metrics
from sklearn.metrics import (r2_score, mean_squared_error)
#variance_inflation_factor
from statsmodels.stats.outliers_influence import variance_inflation_factor

housing= cast(Bunch, fetch_california_housing())

#spliting data into test and train 
"""
1. take housing data and create a dataframe
2. select to only have "AevRooms" column as x 
3. y value median house values
4. split the data sets into test and train 20/80
"""

df = pd.DataFrame(housing.data, columns=housing.feature_names)
#select column and is 2D
#df[["column1", "column2"]] -> gives columns
X = df

#corr() calculates the correlation coefficient between columns.
#Pandas computes the correlation between every pair of numeric columns and returns a matrix.
corr_matrix = X.corr()

"""
           MedInc HouseAge AveRooms
MedInc      1.00    -0.12    0.32
HouseAge   -0.12     1.00   -0.15
AveRooms    0.32    -0.15    1.00

Diagonal is always: because a feature is perfectly correlated with itself.
"""

high_corr_feature = []

for col1 in corr_matrix.columns:
    for col2 in corr_matrix.columns:
        corr_value = cast(float, corr_matrix.loc[col1, col2])

        if col1 != col2 and abs(corr_value) > 0.8:
            high_corr_feature.append((col1, col2, corr_value))

#creating a dataframe for information 
collinearity_df = pd.DataFrame(high_corr_feature, columns=["feature1", "feature2", "correlation"])
print("\nHighly Correlated Features:\n", collinearity_df)

#VIF (Variance Inflation Factor) measures how much a feature can be predicted from the other features.
"""
calculate: VIF of AveRooms based on how well AveRooms can be explained by the other features.
"""
vif_data = pd.DataFrame()
#adding coulmn for names 
vif_data["features"] = X.columns
#calculate VIF
"""
X.values() -> It converts the entire DataFrame into a 2D NumPy array. takes rows not columns
returns:
array([
    [8.3, 41, 6.9],
    [7.2, 21, 6.2],
    [5.6, 30, 5.8]
])
"""
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]
# Print VIF values.
print("\nVariance Inflation Factor (VIF) for each feature:\n", vif_data)




#Dropping AveCedrms to improve performance
X = X = pd.DataFrame(housing.data, columns=housing.feature_names).drop(columns=["AveBedrms"])

y = housing.target
#inserting data into variables
#random_state->42 so shuffling gives same output
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
Why scale?
To put your features (X) onto a similar numerical scale so the ML algorithm can work with them more effectively.
Xscaled = x - mean / std_deviation

fit() = learn the scaling information (mean & std_deviation)
transform() = apply that information
fit_transform() = learn + apply in one step

why only transform for X_test?
Because the test set is supposed to represent unseen data.

we use same data to transform X_test and X_train -> cause we want both to speak same language
"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

##Instantiate linear regression model
model = LinearRegression()
#training the train data
model.fit(X=X_train_scaled, y=y_train)


##making predictions
y_pred = model.predict(X_test_scaled)

"""
# MSE (Mean Squared Error)
# Measures the average squared difference between actual and predicted values.
# Lower is better. Perfect predictions = 0.

# RMSE (Root Mean Squared Error)
# Square root of MSE.
# Easier to interpret because it is in the same units as the target variable.
# Lower is better. Perfect predictions = 0.

# R² Score (Coefficient of Determination)
# Measures how much of the target's variation is explained by the model.
# 1.0 = perfect fit
# 0.0 = no better than predicting the mean
# < 0 = worse than predicting the mean
# Higher is better.
"""

"""
# Mean Squared Error (MSE)
# MSE = (1/n) * Σ(y_true - y_pred)^2

# Root Mean Squared Error (RMSE)
# RMSE = √MSE

# R² Score (Coefficient of Determination)
# R² = 1 - (RSS / TSS)

# RSS (Residual Sum of Squares)
# RSS = Σ(y_true - y_pred)^2

# TSS (Total Sum of Squares)
# TSS = Σ(y_true - y_mean)^2
"""

# R^2 score
R2 = r2_score(y_true=y_test, y_pred=y_pred)
print(f"R^2 score : {R2:.4f}")

#MSE score

MSE = mean_squared_error(y_true=y_test, y_pred=y_pred)
print(f"MSE: {MSE:.4f}")

RMSE =  MSE**0.5
print(f"SMSE: {RMSE:.4f}")

"""
printing the intercept and coefficients
"""
print("Intercept", model.intercept_)

coeff_df = pd.DataFrame({"Features":X.columns, "Coefficient":model.coef_})
print("\nFeature Coefficients:\n", coeff_df)



"""
Graphing the insights
"""
# Sort dataframe by coefficients.
coef_df_sorted = coeff_df.sort_values(by="Coefficient", ascending=False)

#ploting graph
plt.figure(figsize=(8, 6))
plt.barh(
    coef_df_sorted["Features"],
    coef_df_sorted["Coefficient"]
)
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.title("Feature Importance (Linear Regression Coefficients)")
plt.show()


#compute residuals 
residual = y_test - y_pred

plt.figure(figsize=(12, 5))

# Plot 1: Residuals Distribution.
plt.subplot(1, 2, 1)

sns.histplot(residual, bins=30, kde=True, color="blue")

# cross create
plt.axvline(x=0, color="red", linestyle="-")

plt.title("Residuals Distribution")
plt.xlabel("Residuals (y_actual - y_predicted)")
plt.ylabel("Frequency")


# Plot 2: Regression Fit (Actual vs Predicted).
plt.subplot(1, 2, 2)

#alpha -> controls transpirancy
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)

#plt.plot(x_values, y_values)
plt.plot([min(y_test), max(y_test)], min(y_test), max(y_test), color="red", linestyle="-")

plt.xlabel("Actual Prices (in $100,000s)")
plt.ylabel("Predicted Prices (in $100,000s)")

plt.tight_layout()
plt.show()


