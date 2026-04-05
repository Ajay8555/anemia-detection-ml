import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pickle

# ---------------- LOAD DATA ----------------
df = pd.read_csv("CBC_data_for_meandeley_csv.csv")
df.columns = df.columns.str.strip()

# ---------------- CLEAN ----------------
df["Gender"] = df["Gender"].astype(int)
df["Target"] = df["Target"].astype(int)

# ---------------- FEATURES (WITH HGB) ----------------
features = ["Age","Gender","RBC","PCV","MCV","MCH","MCHC","RDW","WBC","PLT","HGB"]

X = df[features]
y = df["Target"]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- SCALE (ONLY FOR SVM) ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- MODELS ----------------
models = {
    "random_forest": RandomForestClassifier(n_estimators=200),
    "svm": SVC(probability=True),
    "naive_bayes": GaussianNB(),
    "adaboost": AdaBoostClassifier(n_estimators=200),
    "xgboost": XGBClassifier(eval_metric='logloss')
}

# ---------------- TRAIN & SAVE ----------------
for name, model in models.items():
    print(f"Training {name}...")

    if name == "svm":
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)

    with open(f"{name}_with_hgb.pkl", "wb") as f:
        pickle.dump(model, f)

print("✅ All models trained WITH HGB")

print("\n📊 DATA SPLIT INFO:")
print("Total rows:", len(df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))