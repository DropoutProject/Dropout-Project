from flask import Flask, request, jsonify, render_template
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

    # GPA check
    gpa = float(student_data.get('GPA', 4.0))
    if gpa < 2.0:
        suggestions.append('📘 Improve academic performance: <a href="https://www.khanacademy.org/" target="_blank">Khan Academy</a> or <a href="https://www.coursera.org/" target="_blank">Coursera</a>')

    # Attendance check
    attendance = float(student_data.get('attendance_rate', 100))
    if attendance < 60:
        suggestions.append('📅 Improve attendance: <a href="https://www.mindtools.com/pages/main/newMN_HTE.htm" target="_blank">Time Management Guide</a>')

    # Study hours check
    study_hours = float(student_data.get('study_hours_per_week', 5))
    if study_hours < 1:
        suggestions.append('⏱️ Study more using Pomodoro: <a href="https://pomofocus.io/" target="_blank">Pomofocus</a>')

    # Parental education check
    parental_edu = student_data.get('parental_education', '').lower()
    if parental_edu in ['none', 'primary']:
        suggestions.append('👨‍🏫 Get academic support: <a href="https://www.topuniversities.com/student-info/admissions-advice/how-find-right-mentor" target="_blank">Find a Mentor</a>')

    # Income level check
    income = student_data.get('income_level', '').lower()
    if income == 'low':
        suggestions.append('💰 Explore financial help: <a href="https://www.scholarships.com/" target="_blank">Scholarships.com</a>')

    # Extracurriculars check
    extracurriculars = student_data.get('extracurriculars', 'yes').lower()
    if extracurriculars == 'no':
        suggestions.append('🎯 Join extracurriculars: <a href="https://blog.prepscholar.com/the-complete-list-of-extracurricular-activities" target="_blank">Extracurricular Benefits</a>')

    return suggestions


@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predictor.html")
    
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        df = pd.DataFrame([data])
        X = preprocess_data(df)
        preds, probs = ensemble_predict(xgb_model, nn_model, X)
        
        response_data = {
            "prediction": int(preds[0]),
            "probability": float(probs[0])
        }
        suggestions = []
        
        if preds[0] == 1:
            suggestions = get_tailored_suggestions(data)
            response_data["suggestions"] = suggestions
        
        return jsonify(response_data) if request.is_json else render_template(
            "predictor.html",
            prediction=response_data['prediction'],
            probability=response_data['probability'],
            suggestions=suggestions
        )
    except Exception as e:
        logging.exception("Prediction failed")
        error_response = {"error": str(e)}
        return jsonify(error_response), 500 if request.is_json else render_template("predictor.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)