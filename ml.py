# ================== IMPORT ==================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# ================== 1. LOAD DATA ==================
file_path = r"C:\Users\LOQ\OneDrive\Desktop\Heart.csv"
df = pd.read_csv(file_path)

# Xóa khoảng trắng tên cột
df.columns = df.columns.str.strip()

print("===== DATA HEAD =====")
print(df.head())

# ================== 2. CLEAN DATA ==================
df.replace('?', np.nan, inplace=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(df.median(), inplace=True)

print("\n===== DATA INFO SAU CLEAN =====")
print(df.info())

# ================== 3. XỬ LÝ LABEL ==================
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

# ================== 4. VISUALIZE ==================
# 4.1 Countplot
plt.figure(figsize=(6,4))
ax = sns.countplot(data=df, x='num', hue='num', palette='Pastel1', legend=False)

plt.title("Phân bố bệnh tim")
plt.xlabel("0 = Không bệnh | 1 = Có bệnh")
plt.ylabel("Số lượng")

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 4.2 Correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# ================== 5. TÁCH DATA ==================
X = df.drop("num", axis=1)
y = df["num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================== 6. SCALE ==================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================== 7. MODEL ==================
# Fix overfitting cho Decision Tree
model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)

# Logistic Regression
model_lr = LogisticRegression(max_iter=1000, random_state=42)

# Train
model_dt.fit(X_train, y_train)
model_lr.fit(X_train_scaled, y_train)

# ================== 8. EVALUATE ==================
print("\n===== ACCURACY =====")

y_pred_dt = model_dt.predict(X_test)
print("DecisionTree:", accuracy_score(y_test, y_pred_dt))

y_pred_lr = model_lr.predict(X_test_scaled)
print("LogisticRegression:", accuracy_score(y_test, y_pred_lr))

# ================== 9. CROSS VALIDATION ==================
cv_scores = cross_val_score(model_lr, X_train_scaled, y_train, cv=5)

print("\n===== CROSS VALIDATION =====")
print("CV Accuracy:", cv_scores.mean())

# ================== 10. CONFUSION MATRIX ==================
cm = confusion_matrix(y_test, y_pred_lr)

print("\n===== CONFUSION MATRIX =====")
print(cm)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================== 11. CLASSIFICATION REPORT ==================
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred_lr))

# ================== 12. ROC CURVE ==================
y_prob = model_lr.predict_proba(X_test_scaled)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# ================== 13. TEST 1 NGƯỜI ==================
sample = [63,1,3,145,233,1,0,150,0,2.3,0,0,1]

sample_df = pd.DataFrame([sample], columns=X.columns)
sample_scaled = scaler.transform(sample_df)

pred = model_lr.predict(sample_scaled)

print("\n===== TEST 1 NGƯỜI =====")
print("KẾT QUẢ:", "CÓ BỆNH TIM" if pred[0] == 1 else "KHÔNG BỆNH")

# ================== 14. SAVE MODEL ==================
joblib.dump(model_lr, r"D:\TEXT.MLDL\model.pkl")
joblib.dump(scaler, r"D:\TEXT.MLDL\scaler.pkl")

print("\n✔️ Đã lưu model + scaler")

# ================== 15. TEST NHANH ==================
print("\n===== TEST NHANH =====")

y_pred = model_lr.predict(X_test_scaled)

for i in range(5):
    print(f"Sample {i+1}: Thực tế = {y_test.iloc[i]} | Dự đoán = {y_pred[i]}")