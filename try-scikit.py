# =========================================
# 1. IMPORT LIBRARIES
# =========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# =========================================
# 2. LOAD DATA
# =========================================

df = pd.read_csv("data.csv")

# =========================================
# 3. DATA EXPLORATION (PANDAS)
# =========================================

print("\n--- First 5 rows (head) ---")
print(df.head())

print("\n--- Dataset Info (info) ---")
print(df.info())

print("\n--- Statistical Summary (describe) ---")
print(df.describe())

print("\n--- Shape of dataset ---")
print(df.shape)

print("\n--- Column names ---")
print(df.columns)

print("\n--- Value counts of target (Purchased) ---")
print(df["Purchased"].value_counts())

print("\n--- Missing values check ---")
print(df.isnull().sum())

# =========================================
# 4. FEATURES & TARGET
# =========================================

X = df[["Age", "Salary"]]
y = df["Purchased"]

# =========================================
# 5. TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================================
# 6. CREATE MODEL
# =========================================

model = DecisionTreeClassifier()

# =========================================
# 7. TRAIN MODEL (fit)
# =========================================

print("\nTraining model...")
model.fit(X_train, y_train)

# =========================================
# 8. PREDICTION
# =========================================

y_pred = model.predict(X_test)

# =========================================
# 9. EVALUATION
# =========================================

print("\n--- Accuracy Score ---")
print(accuracy_score(y_test, y_pred))

print("\n--- Model Score ---")
print(model.score(X_test, y_test))

# =========================================
# 10. CUSTOM PREDICTION (FIXED WARNING)
# =========================================

sample = pd.DataFrame([[25, 50000]], columns=["Age", "Salary"])

prediction = model.predict(sample)

print("\n--- Custom Prediction ---")
print("Result:", prediction)
