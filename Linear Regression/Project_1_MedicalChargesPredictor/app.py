from flask import Flask, render_template, request
import pickle

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html', prediction=None)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        age = request.form.get('age')
        bmi = request.form.get('bmi')
        nchild = request.form.get('children')

        # Convert inputs to numerical values
        age = int(age) if age else 0
        bmi = float(bmi) if bmi else 0.0
        nchild = int(nchild) if nchild else 0

        # Make prediction
        prediction = model.predict([[age, bmi, nchild]])

        # Return template with prediction result
        return render_template('home.html', prediction=int(round(prediction[0])))

    return render_template('home.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

