from flask import Flask, render_template, request
import pickle

model = pickle.load(open('iri.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if(request.method=='POST'):
        thalassemia = int(request.form.get('thalassemia'))
        previousHeartAttack = int(request.form.get('previousHeartAttack'))
        chestPain = int(request.form.get('chestPain'))
        ethnicity = int(request.form.get('ethnicity'))
        stressLevel = int(request.form.get('stressLevel'))
        medication = int(request.form.get('medication'))
        gender = int(request.form.get('gender'))
        alcohol = int(request.form.get('alcohol'))


if(__name__ == '__main__'):
    app.run(debug=True)