from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ================== LOAD MODEL ==================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

columns = ["age","sex","cp","trestbps","chol","fbs","restecg",
           "thalach","exang","oldpeak","slope","ca","thal"]

# ================== ROUTE ==================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        try:
            data = [float(request.form[col]) for col in columns]

            df = pd.DataFrame([data], columns=columns)
            scaled = scaler.transform(df)

            pred = model.predict(scaled)[0]

            result = "CÓ BỆNH TIM" if pred == 1 else "KHÔNG BỆNH"

        except Exception as e:
            result = f"Lỗi input: {str(e)}"

    return render_template("index.html", result=result, columns=columns)

# ================== RUN (LOCAL ONLY) ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)