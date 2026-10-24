"""
Train and save one regression model and one classification model.
Run: python train_models.py
"""
from pathlib import Path
import joblib
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# 1. Regression: predict a numeric diabetes-progression measure.
diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
reg_predictions = regression_model.predict(X_test)
regression_r2 = r2_score(y_test, reg_predictions)

joblib.dump(regression_model, MODEL_DIR / "regression_model.joblib")
joblib.dump(diabetes.feature_names, MODEL_DIR / "regression_features.joblib")

# 2. Classification: predict one of the three Iris flower species.
iris = load_iris()
X_clf, y_clf = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

classification_model = LogisticRegression(max_iter=1000)
classification_model.fit(X_train, y_train)
clf_predictions = classification_model.predict(X_test)
classification_accuracy = accuracy_score(y_test, clf_predictions)

joblib.dump(classification_model, MODEL_DIR / "classification_model.joblib")
joblib.dump(iris.feature_names, MODEL_DIR / "classification_features.joblib")
joblib.dump(iris.target_names, MODEL_DIR / "iris_target_names.joblib")

print("Models trained and saved in the 'models' folder.")
print(f"Regression R²: {regression_r2:.4f}")
print(f"Classification accuracy: {classification_accuracy:.4f} "
      f"({classification_accuracy * 100:.2f}%)")
