from flask import Flask, render_template, redirect, request
import pickle
import numpy as np

model = pickle.load(open('iri.pkl','rb'))

app= Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    locations = ['Adelaide', 'Albany', 'Albury', 'AliceSprings', 'BadgerysCreek',
        'Ballarat', 'Bendigo', 'Brisbane', 'Cairns', 'Canberra', 'Cobar',
        'CoffsHarbour', 'Dartmoor', 'Darwin', 'GoldCoast', 'Hobart',
        'Katherine', 'Launceston', 'Melbourne', 'MelbourneAirport',
        'Mildura', 'Moree', 'MountGambier', 'MountGinini', 'Newcastle',
        'Nhil', 'NorahHead', 'NorfolkIsland', 'Nuriootpa', 'PearceRAAF',
        'Penrith', 'Perth', 'PerthAirport', 'Portland', 'Richmond', 'Sale',
        'SalmonGums', 'Sydney', 'SydneyAirport', 'Townsville',
        'Tuggeranong', 'Uluru', 'WaggaWagga', 'Walpole', 'Watsonia',
        'Williamtown', 'Witchcliffe', 'Wollongong', 'Woomera']
    directions = ['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
        'SSW', 'SW', 'W', 'WNW', 'WSW', 'nan']
    rt = ['Yes', 'No']
    prediction = "No data entered"
    if request.method=='POST':
        form_data = request.form
        input_data = []
        input_data.append(float(form_data['MinTemp']))
        input_data.append(float(form_data['MaxTemp']))
        input_data.append(float(form_data['Rainfall']))
        input_data.append(float(form_data['Evaporation']))
        input_data.append(float(form_data['Sunshine']))
        input_data.append(float(form_data['WindGustSpeed']))
        input_data.append(float(form_data['WindSpeed9am']))
        input_data.append(float(form_data['WindSpeed3pm']))
        input_data.append(float(form_data['Humidity9am']))
        input_data.append(float(form_data['Humidity3pm']))
        input_data.append(float(form_data['Pressure9am']))
        input_data.append(float(form_data['Pressure3pm']))
        input_data.append(float(form_data['Cloud9am']))
        input_data.append(float(form_data['Cloud3pm']))
        input_data.append(float(form_data['Temp9am']))
        input_data.append(float(form_data['Temp3pm']))

        input_data += [1 if form_data['Location'] == loc else 0 for loc in locations]
        input_data += [1 if form_data['WindGustDir'] == dir else 0 for dir in directions]
        input_data += [1 if form_data['WindDir9am'] == dir else 0 for dir in directions]
        input_data += [1 if form_data['WindDir3pm'] == dir else 0 for dir in directions]
        input_data += [1 if form_data['rainToday'] == val else 0 for val in rt]
        
        input_array = np.array(input_data).reshape(1, -1)
        print(len(input_array[0]))
        prediction = model.predict(input_array)[0]
        print(prediction)
        return render_template('index.html', prediction=prediction)
    return render_template('index.html', prediction=prediction)

if(__name__=='__main__'):
    app.run(debug=True)