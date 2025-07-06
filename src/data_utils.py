import pandas as pd
import numpy as np
import torch as pt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def clean_depression_data():
    file_path = 'student_depression_dataset.csv'
    data = pd.read_csv(file_path)

    # Check for missing values
    print("\nMissing values per column:")
    print(data.isnull().sum())

    # Drop incomplete/uninformative columns
    data.drop(columns=['Job Satisfaction', 'Work Pressure', 'id', 'Profession'], inplace=True, errors='ignore')

    # Handle missing values (e.g., drop rows with missing target or fill missing features)
    data.replace('?', np.nan, inplace=True)  # Replace '?' with NaN
    data.dropna(subset=['Depression'], inplace=True)  # Drop rows where the target is missing
    data.fillna(data.median(numeric_only=True), inplace=True)  # Fill missing numerical values with median
    data.fillna(data.mode().iloc[0], inplace=True)  # Fill missing categorical values with mode

    # Binary columns
    binary_columns = ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    for col in binary_columns:
        data[col] = data[col].map({'No': 0, 'Yes': 1})

    # Encode categorical variables
    categorical_columns = ['Gender', 'City', 'Sleep Duration', 'Dietary Habits', 'Degree']  # Update with actual categorical columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True, dtype=int)


    # Normalize numerical features
    numerical_columns = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress']  # Update with actual numerical columns
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Separate features and target
    X = data.drop(columns=['Depression'])  # Features
    y = data['Depression']  # Target

    return X, y

def preprocess_data(X, y):

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split into training and development sets
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Process the training set
    X_train = pt.tensor(X_train.values, requires_grad=False, dtype=pt.float32)
    y_train = pt.tensor(y_train.values, requires_grad=False, dtype=pt.long)

    # Process the testing set
    X_test = pt.tensor(X_test.values, requires_grad=False, dtype=pt.float32)
    y_test = pt.tensor(y_test.values, requires_grad=False, dtype=pt.long)

    # Process the development set
    X_dev = pt.tensor(X_dev.values, requires_grad=False, dtype=pt.float32)
    y_dev = pt.tensor(y_dev.values, requires_grad=False, dtype=pt.long)

    return X_train, y_train, X_test, y_test, X_dev, y_dev



