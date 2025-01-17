# Tourism Activity Profit Maximization

This project analyzes and ranks tourism activities to maximize profit using machine learning algorithms. The dataset includes features such as visitor count, category, and revenue, enabling us to predict and rank categories based on their profit per visitor.

## Dataset

The dataset used for this project is sourced from [Kaggle Tourism Dataset](https://www.kaggle.com/datasets/umeradnaan/tourism-dataset).  
Key features in the dataset:
- **Location**: Unique ID for location.
- **Country**: The country where the activity is located.
- **Category**: Type of activity (e.g., Nature, Cultural, Beach).
- **Visitors**: Annual visitor count.
- **Rating**: Average activity rating.
- **Revenue**: Total revenue generated.
- **Accommodation_Available**: Whether accommodation is available (Yes/No).

---

## Algorithms Used

### 1. **K-Nearest Neighbors (K-NN)**  
A regression algorithm used to predict profit per visitor by considering similar activities.

### 2. **AdaBoost Regressor**  
An ensemble learning method that boosts weak predictors for better accuracy in ranking and profit prediction.
