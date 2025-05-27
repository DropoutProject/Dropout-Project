import requests
import json

# Define the URL for the /predict route
url = "http://127.0.0.1:5000/predict"

# Create a sample payload to send in the POST request
data = {
    "GPA": 3.5,
    "age": 18,
    "income_level": "middle",
    "extracurriculars": 2,
    "parental_education": "bachelor",
    "study_hours_per_week": 15,
    "attendance_rate": 0.9,
    "failed_courses": 0
}
# Send a POST request to the Flask app
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    print("Prediction response:", response.json())  # Output prediction result
else:
    print(f"Error: {response.status_code}")
