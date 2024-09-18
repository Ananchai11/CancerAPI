from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("The_Cancer.pkl")

@app.route('/api/cancer', methods=['POST'])
def cancer():
    
    age = int(request.form.get('age'))
    gender = int(request.form.get('gender'))
    bmi = float(request.form.get('bmi'))
    smoking = int(request.form.get('smoking'))
    geneticrisk = int(request.form.get('geneticrisk'))
    physicalactivity = float(request.form.get('physicalactivity'))  
    alcoholIntake = float(request.form.get('alcoholIntake'))
    cancerhistory = int(request.form.get('cancerhistory')) 
    
    # Prepare the input for the model
    x = np.array([[gender, age, bmi, smoking, geneticrisk, physicalactivity, alcoholIntake, cancerhistory]])

    # Predict using the model
    prediction = model.predict(x)
    if int(prediction[0] == 0):
        return {'prediction': 'ไม่เป็นมะเร็ง'}
    else :
        return {'prediction': 'เป็นมะเร็ง'}
    # Return the result
    #return {'ผลการคาดการ': int(round(prediction[0], 2))}, 200   

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)