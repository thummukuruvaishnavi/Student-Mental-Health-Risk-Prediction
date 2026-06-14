import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# -------------------------
# LOAD DATASET
# -------------------------
data = pd.read_csv("student_mental_health_500.csv")

X = data.drop("risk", axis=1)
y = data["risk"]

# -------------------------
# SPLIT DATA
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# FEATURE SCALING
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# 10 MODELS
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

# -------------------------
# TRAIN & EVALUATE
# -------------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)

print("\n===== COMPARISON TABLE =====\n")
print(results_df)

# -------------------------
# GRAPH COMPARISON (ACCURACY)
# -------------------------
plt.figure()
plt.bar(results_df["Algorithm"], results_df["Accuracy"])
plt.xticks(rotation=45, ha="right")
plt.title("Accuracy Comparison of 10 Algorithms")
plt.ylabel("Accuracy")
plt.xlabel("Algorithm")
plt.tight_layout()
plt.show()
