# I am importing the required libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -----------------------------------------
# 1. LOAD DATA
# -----------------------------------------

df = pd.read_csv("data.csv")

print("First look at the data:")
print(df.head())

# -----------------------------------------
# 2. SPLIT FEATURES AND TARGET
# -----------------------------------------

# Features (input)
X = df[["Age", "Salary"]]

# Target (output)
y = df["Purchased"]

# -----------------------------------------
# 3. SPLIT TRAINING AND TESTING DATA
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------------
# 4. CREATE MODEL
# -----------------------------------------

model = DecisionTreeClassifier()

# -----------------------------------------
# 5. TRAIN MODEL
# -----------------------------------------

print("\nTraining model...")
model.fit(X_train, y_train)

# -----------------------------------------
# 6. MAKE PREDICTIONS
# -----------------------------------------

predictions = model.predict(X_test)

# -----------------------------------------
# 7. EVALUATE MODEL
# -----------------------------------------

print("\nAccuracy Score:")
print(accuracy_score(y_test, predictions))

print("\nModel Score:")
print(model.score(X_test, y_test))

# -----------------------------------------
# 8. TEST CUSTOM INPUT (IMPORTANT FIX)
# -----------------------------------------

# IMPORTANT: use DataFrame to avoid warning

sample = pd.DataFrame([[25, 50000]], columns=["Age", "Salary"])

result = model.predict(sample)

print("\nPrediction for custom input (Age=25, Salary=50000):")
print(result)
