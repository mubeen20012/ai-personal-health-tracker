from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from prediction_pipeline import predict_patient
import numpy as np

app = Flask(__name__)

# -----------------------------
# Upload folder
# -----------------------------
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------
# Routes
# -----------------------------
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        try:
            # --- Tabular Inputs ---
            tabular_input = {
                "Age": float(request.form['Age']),
                "BMI": float(request.form['BMI']),
                "Sex_Male": 1 if request.form['Gender'] == 'Male' else 0,
                "RestingBP": float(request.form['RestingBP']),
                "Cholesterol": float(request.form['Cholesterol'])
            }

            # --- Handle ECG Image ---
            ecg_file = request.files.get('ecg_image', None)
            ecg_path = None
            if ecg_file and allowed_file(ecg_file.filename):
                ecg_filename = secure_filename(ecg_file.filename)
                ecg_path = os.path.join(app.config['UPLOAD_FOLDER'], ecg_filename)
                ecg_file.save(ecg_path)

            # --- Handle X-ray Image ---
            xray_file = request.files.get('xray_image', None)
            xray_path = None
            if xray_file and allowed_file(xray_file.filename):
                xray_filename = secure_filename(xray_file.filename)
                xray_path = os.path.join(app.config['UPLOAD_FOLDER'], xray_filename)
                xray_file.save(xray_path)

            # --- Predict ---
            # If ECG as image, you can send a placeholder numeric array if your model needs numbers
            ecg_input = np.zeros(187)  # dummy array for existing model

            results_df, _ = predict_patient(tabular_input, ecg_input, xray_path)
            result_dict = results_df.iloc[0].to_dict()
            # Pass uploaded paths to template
            result_dict['xray_path'] = xray_file.filename if xray_file else None
            result_dict['ecg_path'] = ecg_file.filename if ecg_file else None

            return render_template('result.html', result=result_dict)

        except Exception as e:
            return f"Error: {e}"

    return render_template('index.html')

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
