
from flask import Flask, request, jsonify
import pandas as pd
from utils.model_loader import load_models
from utils.ensemble import ensemble_predict
from utils.preprocess import preprocess_data
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
xgb_model, nn_model = load_models()

def get_tailored_suggestions(student_data):
    suggestions = []

    if student_data.get('GPA', 4.0) < 2.0:
        suggestions.append(
            "📘 Improve academic performance: [Khan Academy](https://www.khanacademy.org/) or [Coursera](https://www.coursera.org/)"
        )
    if student_data.get('Attendance', 100) < 60:
        suggestions.append(
            "📅 Improve attendance: [Time Management Guide](https://www.mindtools.com/pages/main/newMN_HTE.htm)"
        )
    if student_data.get('StudyHours', 5) < 1:
        suggestions.append(
            "⏱️ Study more using Pomodoro: [Pomofocus](https://pomofocus.io/)"
        )
    if student_data.get('ParentalEducation', '').lower() == 'low':
        suggestions.append(
            "👨‍🏫 Get academic support: [Find a Mentor](https://www.topuniversities.com/student-info/admissions-advice/how-find-right-mentor)"
        )
    if student_data.get('IncomeLevel', '').lower() == 'low':
        suggestions.append(
            "💰 Explore financial help: [Scholarships.com](https://www.scholarships.com/)"
        )
    if student_data.get('Extracurriculars', 1) == 0:
        suggestions.append(
            "🎯 Join extracurriculars: [Extracurricular Benefits](https://blog.prepscholar.com/the-complete-list-of-extracurricular-activities)"
        )

    return suggestions

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        X = preprocess_data(df)
        preds, probs = ensemble_predict(xgb_model, nn_model, X)
        response = {"prediction": int(preds[0]), "probability": float(probs[0])}
        if preds[0] == 1:
            response["suggestions"] = get_tailored_suggestions(data)
        return jsonify(response)
    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
