import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv('./data/tourism_dataset.csv')

# Data preprocessing
# Encoding categorical data
label_encoder = LabelEncoder()
data['Country'] = label_encoder.fit_transform(data['Country'])
data['Category'] = label_encoder.fit_transform(data['Category'])
data['Accommodation_Available'] = label_encoder.fit_transform(data['Accommodation_Available'])

# Feature selection and scaling
features = ['Country', 'Category', 'Visitors', 'Rating', 'Accommodation_Available']
X = data[features]
y = data['Revenue'] / data['Visitors']  # Profit per visitor

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# K-NN Regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)

# AdaBoost Regressor
ada = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4),n_estimators=100,learning_rate=0.1,random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
mse_ada = mean_squared_error(y_test, y_pred_ada)

# Results
print("Mean Squared Error (K-NN):", mse_knn)
print("Mean Squared Error (AdaBoost):", mse_ada)

# Visualization
# Codificarea inițială a categoriilor
label_encoder.fit(data['Category'])

# Verificarea valorilor unice și decodarea lor
unique_categories = [label for label in data['Category'].unique() if label in label_encoder.classes_]

# Obține valorile originale (inverse_transform)
categories = label_encoder.inverse_transform(unique_categories)
print("Categoriile decodificate:", categories)


# Compute average profit per visitor by category using AdaBoost
average_profit = []
for category in unique_categories:
    category_data = X_scaled[data['Category'] == category]
    avg_profit = ada.predict(category_data).mean()
    average_profit.append(avg_profit)

# Bar chart for category ranking
plt.figure(figsize=(10, 6))
plt.bar(categories, average_profit, color='skyblue')
plt.title('Average Profit per Visitor by Category (AdaBoost)', fontsize=14)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Average Profit per Visitor', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('category_profit_ranking.png')
plt.show()

# Visualization of model performance
plt.figure(figsize=(8, 5))

# Bar chart for Mean Squared Error comparison
models = ['K-NN', 'AdaBoost']
mse_values = [mse_knn, mse_ada]

plt.bar(models, mse_values, color=['skyblue', 'orange'])
plt.title('Model Performance: MSE Comparison', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.ylim(0, max(mse_values) * 1.2)  # Extend the y-axis for better visualization
plt.tight_layout()

# Save the graph
plt.savefig('model_performance_comparison.png')
plt.show()

