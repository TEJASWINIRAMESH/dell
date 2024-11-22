# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import numpy as np

# # Load data from CSV file
# df = pd.read_csv(r"C:\Users\Gowsika\Documents\Sem-7\DL LAB\Sem\dlmodel-main\Housing.csv")  # Replace with the actual path to your CSV file

# # Encode categorical variables
# label_encoder = LabelEncoder()
# categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
# for col in categorical_columns:
#     df[col] = label_encoder.fit_transform(df[col])

# # Define features and target variable
# X = df.drop("price", axis=1)
# y = df["price"]

# # Scale numeric features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train Decision Tree model
# dt_regressor = DecisionTreeRegressor(max_depth=10, random_state=42)
# dt_regressor.fit(X_train, y_train)
# dt_predictions = dt_regressor.predict(X_test)

# # Define parameter grid for Random Forest tuning
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 15, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Initialize and fit GridSearchCV for Random Forest
# grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
#                            param_grid=param_grid,
#                            cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')
# grid_search.fit(X_train, y_train)
# best_rf = grid_search.best_estimator_
# rf_predictions = best_rf.predict(X_test)

# # Evaluation Metrics Function
# def get_metrics(y_test, predictions, model_name):
#     mae = mean_absolute_error(y_test, predictions)
#     mse = mean_squared_error(y_test, predictions)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, predictions)
#     explained_var = explained_variance_score(y_test, predictions)
#     return pd.Series({
#         "Model": model_name,
#         "MAE": mae,
#         "MSE": mse,
#         "RMSE": rmse,
#         "R2 Score": r2,
#         "Explained Variance Score": explained_var
#     })

# # Collect scores in DataFrame format
# dt_metrics = get_metrics(y_test, dt_predictions, "Decision Tree Regressor")
# rf_metrics = get_metrics(y_test, rf_predictions, "Tuned Random Forest Regressor")
# metrics_df = pd.DataFrame([dt_metrics, rf_metrics])

# # Display metrics
# print("Model Comparison Table:\n", metrics_df)

# # Show a table with actual vs predicted prices for a sample of the test set
# sample_test_data = X_test[:10]
# sample_actual_prices = y_test[:10].values
# dt_sample_predictions = dt_regressor.predict(sample_test_data)
# rf_sample_predictions = best_rf.predict(sample_test_data)

# # Create a DataFrame for easy visualization
# predictions_df = pd.DataFrame({
#     "Actual Price": sample_actual_prices,
#     "Decision Tree Prediction": dt_sample_predictions,
#     "Tuned Random Forest Prediction": rf_sample_predictions
# })

# print("\nSample Predictions Comparison:\n", predictions_df)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score

dataset = pd.read_csv(r"C:\Users\Gowsika\Documents\Sem-7\DL LAB\Sem\kc_house_data.csv\kc_house_data.csv")


dataset = dataset.drop(['id', 'date'], axis=1)


plt.figure(figsize=(15, 10))
columns = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'floors', 'grade', 'yr_built', 'condition']
sns.heatmap(dataset[columns].corr(), annot=True)

X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(X_train, y_train)
dt_score = dt_regressor.score(X_test, y_test)
dt_predictions = dt_regressor.predict(X_test)
dt_expl_var = explained_variance_score(dt_predictions, y_test)


rf_regressor = RandomForestRegressor(n_estimators=28, random_state=0)
rf_regressor.fit(X_train, y_train)
rf_score = rf_regressor.score(X_test, y_test)
rf_predictions = rf_regressor.predict(X_test)
rf_expl_var = explained_variance_score(rf_predictions, y_test)


def calculate_metrics(model, predictions, y_test, model_name):
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    print(f"{model_name} Metrics:")
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R2 Score:", r2)
    print("\n")
    return mse, mae, rmse, r2


dt_mse, dt_mae, dt_rmse, dt_r2 = calculate_metrics(dt_regressor, dt_predictions, y_test, "Decision Tree Regression")


rf_mse, rf_mae, rf_rmse, rf_r2 = calculate_metrics(rf_regressor, rf_predictions, y_test, "Random Forest Regression")


models_score = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest'],
    'Score': [dt_score, rf_score],
    'Explained Variance Score': [dt_expl_var, rf_expl_var],
    'MSE': [dt_mse, rf_mse],
    'MAE': [dt_mae, rf_mae],
    'RMSE': [dt_rmse, rf_rmse],
    'R2 Score': [dt_r2, rf_r2]
})


models_score.sort_values(by='Score', ascending=False, inplace=True)
print("Model Comparison Table:\n", models_score)

sample_test_data = X_test[:10]
sample_actual_prices = y_test[:10]
dt_sample_predictions = dt_regressor.predict(sample_test_data)
rf_sample_predictions = rf_regressor.predict(sample_test_data)


predictions_df = pd.DataFrame({
    "Actual Price": sample_actual_prices,
    "Decision Tree Prediction": dt_sample_predictions,
    "Random Forest Prediction": rf_sample_predictions
})

print("\nSample Predictions Comparison:\n", predictions_df)
