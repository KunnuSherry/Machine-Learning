import joblib
model_dict = joblib.load('predict.joblib')
model = model_dict['model']
input_columns = ['Store', 'Day', 'Month', 'Year', 'DayOfWeek_1', 'DayOfWeek_2',
       'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6',
       'DayOfWeek_7', 'Promo_0', 'Promo_1', 'StateHoliday_0', 'StateHoliday_a',
       'StateHoliday_b', 'StateHoliday_c', 'StoreType_a', 'StoreType_b',
       'StoreType_c', 'StoreType_d', 'Assortment_a', 'Assortment_b',
       'Assortment_c']

def predict_turnover(store, store_type, assortment, day, month, year, day_of_week, promo, state_holiday):
    input_data = [int(store), int(day), int(month), int(year)]
    for i in range(4,11):
        if input_columns[i] == 'DayOfWeek_'+str(day_of_week):
            input_data.append(1)
        else:
            input_data.append(0)
    if promo == 0:
        input_data.extend([1,0])
    else:
        input_data.extend([0,1])
    # Add StateHoliday_0 check before the loop
    if state_holiday == '0':  # If no holiday, set 'StateHoliday_0' to 1
        input_data.append(1)
        input_data.extend([0, 0, 0])  # Other holidays are 0
    else:
        input_data.append(0)  # No state holiday
        for i in range(13, 16):  # Only iterate over a, b, c (3 values)
            if input_columns[i + 1] == 'StateHoliday_' + state_holiday:  # Shift index
                input_data.append(1)
            else:
                input_data.append(0)
    for i in range(17,20):
        if input_columns[i] == 'StoreType_'+store_type:
            input_data.append(1)
        else:
            input_data.append(0)
    for i in range(20,24):
        if input_columns[i] == 'Assortment_'+assortment:
            input_data.append(1)
        else:
            input_data.append(0)

    return model.predict([input_data])
