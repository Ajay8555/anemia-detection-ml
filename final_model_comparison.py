import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------- LOAD DATA ----------------
df = pd.read_csv("CBC_data_for_meandeley_csv.csv")
df.columns = df.columns.str.strip()

# ---------------- CLEAN ----------------
df["Gender"] = df["Gender"].astype(int)
df["Target"] = df["Target"].astype(int)

# ---------------- FEATURES ----------------
features_with_hgb = ["Age","Gender","RBC","PCV","MCV","MCH","MCHC","RDW","WBC","PLT","HGB"]
features_without_hgb = ["Age","Gender","RBC","PCV","MCV","MCH","MCHC","RDW","WBC","PLT"]

X1 = df[features_with_hgb]
X2 = df[features_without_hgb]
y = df["Target"]

# ---------------- SPLIT ----------------
X1_train, X1_test, y_train, y_test = train_test_split(
    X1, y, test_size=0.2, random_state=42, stratify=y
)

X2_train, X2_test, _, _ = train_test_split(
    X2, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- SCALE (for SVM) ----------------
scaler1 = StandardScaler()
X1_train_scaled = scaler1.fit_transform(X1_train)
X1_test_scaled = scaler1.transform(X1_test)

scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

# ---------------- MODEL FILES ----------------
models_with_hgb = {
    "Random Forest": "random_forest_with_hgb.pkl",
    "SVM": "svm_with_hgb.pkl",
    "Naive Bayes": "naive_bayes_with_hgb.pkl",
    "AdaBoost": "adaboost_with_hgb.pkl",
    "XGBoost": "xgboost_with_hgb.pkl"
}

models_without_hgb = {
    "Random Forest": "random_forest_without_hgb.pkl",
    "SVM": "svm_without_hgb.pkl",
    "Naive Bayes": "naive_bayes_without_hgb.pkl",
    "AdaBoost": "adaboost_without_hgb.pkl",
    "XGBoost": "xgboost_without_hgb.pkl"
}

# ---------------- EVALUATION ----------------
def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0)
    }

# ---------------- RESULTS ----------------
results = []

# WITH HGB
for name, path in models_with_hgb.items():
    with open(path, "rb") as f:
        model = pickle.load(f)

    if name == "SVM":
        metrics = evaluate(model, X1_test_scaled, y_test)
    else:
        metrics = evaluate(model, X1_test, y_test)

    metrics["Model"] = name
    metrics["Scenario"] = "With HGB"
    results.append(metrics)

# WITHOUT HGB
for name, path in models_without_hgb.items():
    with open(path, "rb") as f:
        model = pickle.load(f)

    if name == "SVM":
        metrics = evaluate(model, X2_test_scaled, y_test)
    else:
        metrics = evaluate(model, X2_test, y_test)

    metrics["Model"] = name
    metrics["Scenario"] = "Without HGB"
    results.append(metrics)

# ---------------- FINAL TABLE ----------------
df_results = pd.DataFrame(results)

df_results[["Accuracy","Precision","Recall","F1 Score"]] = df_results[
    ["Accuracy","Precision","Recall","F1 Score"]
].round(4)

print("\n🔥 FINAL MODEL COMPARISON TABLE:\n")
print(df_results)

df_results.to_csv("final_model_comparison.csv", index=False)

print("\n✅ Saved as final_model_comparison.csv")