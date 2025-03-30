from flask import Flask, render_template, request, redirect, url_for
from model import predict_turnover
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        store = request.form.get('store')
        store_type = request.form.get('storeType')
        assortment = request.form.get('assortment')
        day = request.form.get('day')
        month = request.form.get('month')
        year = request.form.get('year')
        day_of_week = request.form.get('dayOfWeek')
        promo = request.form.get('promo')
        state_holiday = request.form.get('stateHoliday')
        prediction = predict_turnover(store, store_type, assortment, day, month, year, day_of_week, promo, state_holiday)
        return render_template('index.html', prediction=prediction[0])
    return render_template('index.html', prediction=-1)



if __name__ == '__main__':
    app.run(debug=True)

