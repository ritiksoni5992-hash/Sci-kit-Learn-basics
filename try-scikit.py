# I am importing the libraries I need
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Now I am loading my dataset
# (Make sure data.csv is in the same folder)
df = pd.read_csv("data.csv")

# Let me quickly see what my data looks like
print("Here are the first few rows of the data:")
print(df.head())

# Checking basic information about the dataset
print("\nLet me see dataset info:")
print(df.info())

# Some basic statistics like mean, min, max, etc.
print("\nStatistical summary of the data:")
print(df.describe())

# Just checking shape (rows and columns)
print("\nShape of dataset:")
print(df.shape)

# Column names
print("\nColumns in dataset:")
print(df.columns)

# Checking if there are any missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Now I will separate features and target
# Features (X) are input columns
# Target (y) is what I want to predict

X = df[["Age", "Salary"]]
y = df["Purchased"]

# Now I split the data into training and testing sets
# 80% training, 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# I am using Decision Tree model for this example
model = DecisionTreeClassifier()

# Now I am training the model
print("\nTraining the model...")
model.fit(X_train, y_train)

# Now I will make predictions
print("\nMaking predictions...")
predictions = model.predict(X_test)

# Let's check how good the model is
print("\nModel Accuracy:")
print(accuracy_score(y_test, predictions))

# Another way to check accuracy
print("\nModel Score:")
print(model.score(X_test, y_test))

# Finally, let's test with a custom input
# Example: Age = 25, Salary = 50000

sample = [[25, 50000]]

result = model.predict(sample)

print("\nPrediction for custom input:")
print(result)
