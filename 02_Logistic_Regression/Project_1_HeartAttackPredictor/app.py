from flask import Flask, render_template, request
import pickle

# Load the trained model
model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', prediction="")

@app.route('/predict', methods=['POST'])
def predict():
    prediction = ""

    if request.method == 'POST':
        try:
            # Get form values with default values
            thalassemia = int(request.form.get('thalassemia', 0))
            previousHeartAttack = int(request.form.get('previousHeartAttack', 0))
            chestPain = int(request.form.get('chestPain', 0))
            ethnicity = int(request.form.get('ethnicity', 0))
            stressLevel = int(request.form.get('stressLevel', 1))  # Default as lowest
            medication = int(request.form.get('medication', 0))
            cholesterol = int(request.form.get('cholesterol', 194))  # Default as median value
            gender = int(request.form.get('gender', 0))
            alcohol = int(request.form.get('alcohol', 1))

            # Make prediction
            predictionint = model.predict([[thalassemia, previousHeartAttack, chestPain, ethnicity, 
                                            stressLevel, medication, cholesterol, gender, alcohol]])[0]

            # Assign message based on prediction
            prediction = "No Risk !" 
            if predictionint == 0:
                prediction = "Risk !"
        
        except ValueError as e:
            prediction = f"Error: Invalid input - {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
