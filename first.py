import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Feature engineering
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Create family size feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Select features for the model
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']
X = df[features].copy()
y = df['Survived']

# Encode categorical variables
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Training Accuracy: {train_score:.4f}")
print(f"Testing Accuracy: {test_score:.4f}")

# Save model and scaler
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }, f)

print("\nModel saved as 'titanic_model.pkl'")
print(f"Feature order: {list(X.columns)}")

# Example prediction
print("\n--- Example Usage ---")
example_input = {
    'Pclass': 3,
    'Sex': 0,  # 0=male, 1=female
    'Age': 22,
    'Fare': 7.25,
    'FamilySize': 1,
    'Embarked_Q': 0,
    'Embarked_S': 1
}
example_df = pd.DataFrame([example_input])
example_scaled = scaler.transform(example_df)
prediction = model.predict(example_scaled)[0]
probability = model.predict_proba(example_scaled)[0]

print(f"Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")
print(f"Probability: {probability[1]:.2%} chance of survival")
