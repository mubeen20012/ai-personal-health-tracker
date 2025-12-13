from flask import Flask, render_template, request
import os
import numpy as np
from werkzeug.utils import secure_filename
from prediction_pipeline import predict_patient

app = Flask(__name__)

# -----------------------------
# Upload folder for X-ray images
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
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # --- Read Tabular Inputs ---
            tabular_input = {
                "Age": float(request.form['Age']),
                "BMI": float(request.form['BMI']),
                "Sex_Male": 1 if request.form['Gender'] == 'Male' else 0,
                "RestingBP": float(request.form['RestingBP']),
                "Cholesterol": float(request.form['Cholesterol'])
            }

            # --- Read ECG Input ---
            ecg_text = request.form.get('ecg_signal', '')
            ecg_input = np.array([float(x.strip()) for x in ecg_text.split(',') if x.strip() != ''])

            # --- Handle X-ray File ---
            file = request.files.get('xray_image', None)
            xray_path = None
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                xray_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(xray_path)

            # --- Predict ---
            results_df, _ = predict_patient(tabular_input, ecg_input, xray_path)
            result_dict = results_df.iloc[0].to_dict()

            return render_template('result.html', result=result_dict)

        except Exception as e:
            return f"Error: {e}"

    return render_template('index.html')

# -----------------------------
# Run the app
# -----------------------------
if __name__ == '__main__':
    # Render sets the PORT environment variable automatically
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
