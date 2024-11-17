import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

from imblearn.over_sampling import SMOTE

import shap

import warnings
warnings.filterwarnings('ignore')

# Data Loading
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("First 5 rows:")
print(df.head())

#  Data Exploration and EDA
print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing)

# Handle missing 'TotalCharges'
df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
print("\nAfter handling missing 'TotalCharges':")
print(df.isnull().sum())

# Churn Distribution
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Churn by Gender
sns.countplot(x='gender', hue='Churn', data=df)
plt.title('Churn by Gender')
plt.show()

# Churn by Contract Type
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type')
plt.show()

# Correlation Heatmap
numerical = ['tenure', 'MonthlyCharges', 'TotalCharges']
corr = df[numerical].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Data Cleaning and Preprocessing
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('customerID')  # Exclude 'customerID' as it's unique

binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

multi_cols = [col for col in categorical_cols if df[col].nunique() > 2]
df = pd.get_dummies(df, columns=multi_cols)



missing_churn = df['Churn'].isnull().sum()
print(f"\nMissing values in 'Churn': {missing_churn}")


if missing_churn > 0:
    df = df.dropna(subset=['Churn'])
    print(f"Dropped rows with missing 'Churn'. New shape: {df.shape}")

unique_churn = df['Churn'].unique()
print(f"Unique values in 'Churn': {unique_churn}")


print(f"Unique values in encoded 'Churn': {df['Churn'].unique()}")

if set(df['Churn'].unique()) == {'No', 'Yes'}:
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    print(f"Unique values in encoded 'Churn' after mapping: {df['Churn'].unique()}")

# Verify encoding
churn_nan = df['Churn'].isnull().sum()
print(f"Missing values in encoded 'Churn': {churn_nan}")

# If encoding introduced NaNs, drop those rows
if churn_nan > 0:
    df = df.dropna(subset=['Churn'])
    print(f"Dropped rows with NaN in 'Churn' after encoding. New shape: {df.shape}")

# Feature Scaling
scaler = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Feature Engineering
df['TenureGroup'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 36, 48, 60, 72], labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
df = pd.get_dummies(df, columns=['TenureGroup'], drop_first=True)

# Model Building
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Check for any remaining NaNs in X or y
print(f"\nMissing values in features (X): {X.isnull().sum().sum()}")
print(f"Missing values in target (y): {y.isnull().sum()}")

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f'Training samples: {X_train.shape[0]}')
print(f'Testing samples: {X_test.shape[0]}')

#  Handle Class Imbalance with SMOTE
print("\nClass distribution before SMOTE:")
print(y_train.value_counts())

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE:")
print(y_train_res.value_counts())

# Initialize and Train Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train models
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    print(f'{name} trained.')

# Model Evaluation
results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    acc = (y_pred == y_test).mean()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    results = results._append({
        'Model': name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc
    }, ignore_index=True)

print("\nModel Evaluation Results:")
print(results)

# Model Interpretation with SHAP
best_model_name = results.sort_values('ROC-AUC', ascending=False).iloc[0]['Model']
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name}")

# Initialize SHAP explainer
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test, plot_type="bar")


# ROC Curve
y_proba_best = best_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba_best)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {roc_auc_score(y_test, y_proba_best):.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, best_model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
