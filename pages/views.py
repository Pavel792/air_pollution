from django.shortcuts import render
import csv
import os
from django.conf import settings
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

MONTHS_RU = ['янв', 'фев', 'мар', 'апр', 'мая', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']

def get_pm25_verdict(value):
    try:
        v = float(value)
    except (ValueError, TypeError):
        return '—'
    if v <= 15:
        return 'Хорошо'
    elif v <= 35:
        return 'Нормально'
    else:
        return 'Плохо'

def get_pm10_verdict(value):
    try:
        v = float(value)
    except (ValueError, TypeError):
        return '—'
    if v <= 30:
        return 'Хорошо'
    elif v <= 50:
        return 'Нормально'
    else:
        return 'Плохо'

def get_so2_verdict(value):
    try:
        v = float(value)
    except (ValueError, TypeError):
        return '—'
    if v <= 50:
        return 'Хорошо'
    elif v <= 125:
        return 'Нормально'
    else:
        return 'Плохо'

def get_co_verdict(value):
    try:
        v = float(value)
    except (ValueError, TypeError):
        return '—'
    if v <= 500:
        return 'Хорошо'
    elif v <= 1000:
        return 'Нормально'
    else:
        return 'Плохо'

def get_no2_verdict(value):
    try:
        v = float(value)
    except (ValueError, TypeError):
        return '—'
    if v <= 40:
        return 'Хорошо'
    elif v <= 100:
        return 'Нормально'
    else:
        return 'Плохо'

def get_o3_verdict(value):
    try:
        v = float(value)
    except (ValueError, TypeError):
        return '—'
    if v <= 100:
        return 'Хорошо'
    elif v <= 160:
        return 'Нормально'
    else:
        return 'Плохо'

def get_day_verdict(row):
    verdicts = []
    total_sum = 0
    try:
        verdicts.append(get_pm25_verdict(row.get('pm25', '0')))
        verdicts.append(get_pm10_verdict(row.get('pm10', '0')))
        verdicts.append(get_so2_verdict(row.get('so2', '0')))
        verdicts.append(get_co_verdict(row.get('co', '0')))
        verdicts.append(get_no2_verdict(row.get('no2', '0')))
    except KeyError:
        return '—'

    for i in verdicts:
        if i == 'Нормально':
            total_sum += 1
        elif i == 'Хорошо':
            total_sum += 2

    return total_sum

def transform_data(data, target):

    df = data.copy()

    df["dayNumber"] = df["date"].dt.dayofyear
    df["monthNumber"] = df["date"].dt.month
    df["dayOfWeek"] = df["date"].dt.dayofweek
    df["isWeekend"] = (df["dayOfWeek"] >= 5).astype(int)

    df["1DayAgo"] = df[target].shift(1)
    df["2DayAgo"] = df[target].shift(2)
    df["3DayAgo"] = df[target].shift(3)
    df["4DayAgo"] = df[target].shift(4)
    df["5DayAgo"] = df[target].shift(5)
    df["6DayAgo"] = df[target].shift(6)
    df["7DayAgo"] = df[target].shift(7)

    df["3DaysMean"] = df[target].rolling(window=3, min_periods=1).mean()
    df["7DaysMean"] = df[target].rolling(window=7, min_periods=1).mean()
    df["3DaysMax"] = df[target].rolling(window=3, min_periods=1).max()
    df["3DaysMin"] = df[target].rolling(window=3, min_periods=1).min()
    df["7DaysStd"] = df[target].rolling(window=7, min_periods=1).std()
    df["trend"] = df[target] - df[target].shift(1)

    return df

def get_forecast(target):
    file_path = os.path.join(settings.BASE_DIR, "data", "pollutants.csv")

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=";")
        rows = list(reader)

    col_ind = {"pm25": 1, "pm10": 2, "co": 3, "so2": 4, "no2": 5}

    data = [[row[0], row[col_ind[target]]] for row in rows[-21:]]

    data = pd.DataFrame(data, columns=["date", target])

    data["date"] = pd.to_datetime(data["date"], format="%d.%m")
    data[target] = data[target].str.replace(",", ".").astype(float)

    if target in ["pm25", "pm10", "so2", "no2"]:
        data[target] = data[target] * 1000

    transformed = transform_data(data, target)
    transformed = transformed.dropna()

    features = [
        "dayNumber", "monthNumber", "dayOfWeek", "isWeekend",
        "1DayAgo", "2DayAgo", "3DayAgo", "4DayAgo",
        "5DayAgo", "6DayAgo", "7DayAgo",
        "3DaysMean", "7DaysMean", "3DaysMax", "3DaysMin",
        "7DaysStd", "trend"
    ]

    model = LinearRegression()
    X_train = transformed[features]
    y_train = transformed[target]
    model.fit(X_train, y_train)

    results = []
    last_values = data[target].values.tolist()
    last_date = data["date"].iloc[-1]

    for day in range(3):
        next_date = last_date + pd.Timedelta(days=1)
        
        day_number = next_date.timetuple().tm_yday
        month_number = next_date.month
        day_of_week = next_date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        if len(last_values) >= 7:
            last_7 = last_values[-7:]
            _1day_ago = last_7[-1]
            _2day_ago = last_7[-2]
            _3day_ago = last_7[-3]
            _4day_ago = last_7[-4]
            _5day_ago = last_7[-5]
            _6day_ago = last_7[-6]
            _7day_ago = last_7[-7]
            _3days_mean = sum(last_7[-3:]) / 3
            _7days_mean = sum(last_7) / 7
            _3days_max = max(last_7[-3:])
            _3days_min = min(last_7[-3:])
            _7days_std = np.std(last_7)
            _trend = last_7[-1] - last_7[-2] if len(last_7) >= 2 else 0
        else:
            _1day_ago = last_values[-1] if last_values else 0
            _2day_ago = last_values[-2] if len(last_values) >= 2 else _1day_ago
            _3day_ago = last_values[-3] if len(last_values) >= 3 else _2day_ago
            _4day_ago = last_values[-4] if len(last_values) >= 4 else _3day_ago
            _5day_ago = last_values[-5] if len(last_values) >= 5 else _4day_ago
            _6day_ago = last_values[-6] if len(last_values) >= 6 else _5day_ago
            _7day_ago = last_values[-7] if len(last_values) >= 7 else _6day_ago
            _3days_mean = sum(last_values[-3:]) / min(3, len(last_values))
            _7days_mean = sum(last_values[-7:]) / min(7, len(last_values))
            _3days_max = max(last_values[-3:])
            _3days_min = min(last_values[-3:])
            _7days_std = np.std(last_values[-7:]) if len(last_values) >= 2 else 0
            _trend = last_values[-1] - last_values[-2] if len(last_values) >= 2 else 0
        
        X_pred = np.array([[
            day_number, month_number, day_of_week, is_weekend,
            _1day_ago, _2day_ago, _3day_ago, _4day_ago,
            _5day_ago, _6day_ago, _7day_ago,
            _3days_mean, _7days_mean, _3days_max, _3days_min,
            _7days_std, _trend
        ]])
        
        pred = model.predict(X_pred)[0]
        results.append(round(pred, 2))
        
        last_values.append(pred)
        last_date = next_date

    return {"day1": results[0], "day2": results[1], "day3": results[2]}

def get_forcast_verdicts(pm25_forecast ,pm10_forecast, so2_forecast, co_forecast, no2_forecast):
    day1_list = [get_pm25_verdict(pm25_forecast["day1"]), get_pm10_verdict(pm10_forecast["day1"]), get_so2_verdict(so2_forecast["day1"]), get_co_verdict(co_forecast["day1"]), get_no2_verdict(no2_forecast["day1"])]
    day2_list = [get_pm25_verdict(pm25_forecast["day2"]), get_pm10_verdict(pm10_forecast["day2"]), get_so2_verdict(so2_forecast["day2"]), get_co_verdict(co_forecast["day2"]), get_no2_verdict(no2_forecast["day2"])]
    day3_list = [get_pm25_verdict(pm25_forecast["day3"]), get_pm10_verdict(pm10_forecast["day3"]), get_so2_verdict(so2_forecast["day3"]), get_co_verdict(co_forecast["day3"]), get_no2_verdict(no2_forecast["day3"])]

    day1_aqi = sum([2 if i == 'Хорошо' else 1 if i == 'Нормально' else 0 for i in day1_list])
    day2_aqi = sum([2 if i == 'Хорошо' else 1 if i == 'Нормально' else 0 for i in day2_list])
    day3_aqi = sum([2 if i == 'Хорошо' else 1 if i == 'Нормально' else 0 for i in day3_list])

    day1_verdict = "-"
    if (day1_aqi <= 4):
        day1_verdict = "плохое"
    else:
        if (day1_aqi >= 8):
            day1_verdict = "хорошее"
        else:
            day1_verdict = "нормальное"
    
    day2_verdict = "-"
    if (day2_aqi <= 4):
        day2_verdict = "плохое"
    else:
        if (day2_aqi >= 8):
            day2_verdict = "хорошее"
        else:
            day2_verdict = "нормальное"

    day3_verdict = "-"
    if (day3_aqi <= 4):
        day3_verdict = "плохое"
    else:
        if (day3_aqi >= 8):
            day3_verdict = "хорошее"
        else:
            day3_verdict = "нормальное"

    return {"day1_aqi" : day1_aqi, "day1_verdict" : day1_verdict, "day2_aqi" : day2_aqi, "day2_verdict" : day2_verdict, "day3_aqi" : day3_aqi, "day3_verdict" : day3_verdict}

def home(request):
    today = datetime.now()
    tomorrow = datetime.now() + timedelta(days=1)
    tomorrow1 = datetime.now() + timedelta(days=2)
    tomorrow2 = datetime.now() + timedelta(days=3)

    PAGES_DIR = Path(__file__).resolve().parent
    FILE_PATH = PAGES_DIR.parent / 'data' / 'pollutants.csv'

    with open(FILE_PATH, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        rows = list(reader)

    today_row = rows[-1]

    pm25_str = today_row[1].strip().replace(',', '.')
    pm25_float = round(float(pm25_str) * 1000, 2)

    pm10_str = today_row[2].strip().replace(',', '.')
    pm10_float = round(float(pm10_str) * 1000, 2)

    CO_str = today_row[3].strip().replace(',', '.')
    CO_float = round(float(CO_str), 2)

    SO2_str = today_row[4].strip().replace(',', '.')
    SO2_float = round(float(SO2_str) * 1000, 2)

    NO2_str = today_row[5].strip().replace(',', '.')
    NO2_float = round(float(NO2_str) * 1000, 2)

    day_pollutants = {
        'pm25': pm25_float,
        'pm10': pm10_float,
        'so2': SO2_float,
        'co': CO_float,
        'no2': NO2_float,
    }
    aqi_sum = get_day_verdict(day_pollutants)
    
    if aqi_sum <= 4:
        quality_text = 'плохое'
        aqi_value = f'{aqi_sum}/10'
    else:
        if aqi_sum >= 8:
            quality_text = 'хорошее'
            aqi_value = f'{aqi_sum}/10'
        else:
            quality_text = 'нормальное'
            aqi_value = f'{aqi_sum}/10'


    pm25_forecast = get_forecast("pm25") 
    pm10_forecast = get_forecast("pm10") 
    so2_forecast = get_forecast("so2") 
    co_forecast = get_forecast("co") 
    no2_forecast = get_forecast("no2") 

    forecast_verdicts = get_forcast_verdicts(pm25_forecast, pm10_forecast, so2_forecast, co_forecast, no2_forecast)

    context = {
        'today': {
            'date': f'{today.day} {MONTHS_RU[today.month - 1]} {today.year}',
            'aqi': aqi_value,
            'quality_verdict': quality_text,
            'pollutants': {
                'pm25': {'value': f'{pm25_float}', 'unit': 'мкг/м³', 'verdict': get_pm25_verdict(pm25_float)},
                'pm10': {'value': f'{pm10_float}', 'unit': 'мкг/м³', 'verdict': get_pm10_verdict(pm10_float)},
                'so2': {'value': f'{SO2_float}', 'unit': 'мкг/м³', 'verdict': get_so2_verdict(SO2_float)},
                'co': {'value': f'{CO_float}', 'unit': 'мг/м³', 'verdict': get_co_verdict(CO_float)},
                'no2': {'value': f'{NO2_float}', 'unit': 'мкг/м³', 'verdict': get_no2_verdict(NO2_float)},
            }
        },
        'predictions': [
            {
                'date': f'{tomorrow.day} {MONTHS_RU[tomorrow.month - 1]} {tomorrow.year}',
                'aqi': f'{forecast_verdicts["day1_aqi"]}/10',
                'quality_verdict': f'{forecast_verdicts["day1_verdict"]}',
                'pollutants': {
                    'pm25': {'value': f'{pm25_forecast["day1"]}', 'unit': 'мкг/м³', 'verdict': get_pm25_verdict(pm25_forecast['day1'])},
                    'pm10': {'value': f'{pm10_forecast["day1"]}', 'unit': 'мкг/м³', 'verdict': get_pm10_verdict(pm10_forecast['day1'])},
                    'so2': {'value': f'{so2_forecast["day1"]}', 'unit': 'мкг/м³', 'verdict': get_so2_verdict(so2_forecast['day1'])},
                    'co': {'value': f'{co_forecast["day1"]}', 'unit': 'мкг/м³', 'verdict': get_co_verdict(co_forecast['day1'])},
                    'no2': {'value': f'{no2_forecast["day1"]}', 'unit': 'мкг/м³', 'verdict': get_no2_verdict(no2_forecast['day1'])}
                }
            },
            {
                'date': f'{tomorrow1.day} {MONTHS_RU[tomorrow1.month - 1]} {tomorrow1.year}',
                'aqi': f'{forecast_verdicts["day2_aqi"]}/10',
                'quality_verdict': f'{forecast_verdicts["day2_verdict"]}',
                'pollutants': {
                    'pm25': {'value': f'{pm25_forecast["day2"]}', 'unit': 'мкг/м³', 'verdict': get_pm25_verdict(pm25_forecast['day2'])},
                    'pm10': {'value': f'{pm10_forecast["day2"]}', 'unit': 'мкг/м³', 'verdict': get_pm10_verdict(pm10_forecast['day2'])},
                    'so2': {'value': f'{so2_forecast["day2"]}', 'unit': 'мкг/м³', 'verdict': get_so2_verdict(so2_forecast['day2'])},
                    'co': {'value': f'{co_forecast["day2"]}', 'unit': 'мкг/м³', 'verdict': get_co_verdict(co_forecast['day2'])},
                    'no2': {'value': f'{no2_forecast["day2"]}', 'unit': 'мкг/м³', 'verdict': get_no2_verdict(no2_forecast['day2'])}
                }
            },
            {
                'date': f'{tomorrow2.day} {MONTHS_RU[tomorrow2.month - 1]} {tomorrow2.year}',
                'aqi': f'{forecast_verdicts["day3_aqi"]}/10',
                'quality_verdict': f'{forecast_verdicts["day3_verdict"]}',
                'pollutants': {
                    'pm25': {'value': f'{pm25_forecast["day3"]}', 'unit': 'мкг/м³', 'verdict': get_pm25_verdict(pm25_forecast['day3'])},
                    'pm10': {'value': f'{pm10_forecast["day3"]}', 'unit': 'мкг/м³', 'verdict': get_pm10_verdict(pm10_forecast['day3'])},
                    'so2': {'value': f'{so2_forecast["day3"]}', 'unit': 'мкг/м³', 'verdict': get_so2_verdict(so2_forecast['day3'])},
                    'co': {'value': f'{co_forecast["day3"]}', 'unit': 'мкг/м³', 'verdict': get_co_verdict(co_forecast['day3'])},
                    'no2': {'value': f'{no2_forecast["day3"]}', 'unit': 'мкг/м³', 'verdict': get_no2_verdict(no2_forecast['day3'])}
                }
            },
        ]
    }
    
    return render(request, 'pages/home.html', context)

def history(request):
    file_path1 = os.path.join(settings.BASE_DIR, "data", "ANALYSED-2024.csv")
    file_path2 = os.path.join(settings.BASE_DIR, "data", "CAMS-gases-2024.csv")
    file_path3 = os.path.join(settings.BASE_DIR, "data", "pollutants.csv")
    file_path4 = os.path.join(settings.BASE_DIR, "data", "predictions.csv")
    
    extra_data = {}
    with open(file_path2, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            extra_data[row['date']] = {
                'co': round(float(row.get('CO', 0)), 2),
                'no2': round(float(row.get('NO2', 0)), 2),
                'so2': round(float(row.get('SO2', 0)), 2),
            }

    predictions_data = {}
    with open(file_path4, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            predictions_data[row['date']] = {
                'pm25': round(float(row.get('PM2.5', 0)) * 1000, 2),
                'pm10': round(float(row.get('PM10', 0)) * 1000, 2),
                'co': round(float(row.get('CO', 0)), 2),
                'so2': round(float(row.get('SO2', 0)) * 1000, 2),
                'no2': round(float(row.get('NO2', 0)) * 1000, 2),
            }

    def add_predictions(row_data, date_obj):
        date_key = date_obj.strftime('%Y-%m-%d')
        pred = predictions_data.get(date_key, {})
        row_data['pred_pm25'] = f"{pred.get('pm25', 0):.2f}"
        row_data['pred_pm10'] = f"{pred.get('pm10', 0):.2f}"
        row_data['pred_co'] = f"{pred.get('co', 0):.2f}"
        row_data['pred_so2'] = f"{pred.get('so2', 0):.2f}"
        row_data['pred_no2'] = f"{pred.get('no2', 0):.2f}"
        row_data['pred_pm25_verdict'] = get_pm25_verdict(pred.get('pm25', 0))
        row_data['pred_pm10_verdict'] = get_pm10_verdict(pred.get('pm10', 0))
        row_data['pred_co_verdict'] = get_co_verdict(pred.get('co', 0))
        row_data['pred_so2_verdict'] = get_so2_verdict(pred.get('so2', 0))
        row_data['pred_no2_verdict'] = get_no2_verdict(pred.get('no2', 0))
        return row_data

    data = []
    
    with open(file_path3, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
        for row in reader:
            if len(row) < 6:
                continue
                
            date_str = row[0].strip()
            try:
                date_obj = datetime.strptime(date_str, '%d.%m')
                date_obj = date_obj.replace(year=datetime.now().year)
            except ValueError:
                continue
                
            pm25 = round(float(row[1].strip().replace(',', '.')) * 1000, 2)
            pm10 = round(float(row[2].strip().replace(',', '.')) * 1000, 2)
            co = round(float(row[3].strip().replace(',', '.')), 2)
            so2 = round(float(row[4].strip().replace(',', '.')) * 1000, 2)
            no2 = round(float(row[5].strip().replace(',', '.')) * 1000, 2)
                
            row_data = {
                'date': f"{date_obj.day} {MONTHS_RU[date_obj.month - 1]} {date_obj.year}",
                'date_obj': date_obj,
                'pm25': f"{pm25:.2f}",
                'pm10': f"{pm10:.2f}",
                'co': f"{co:.2f}",
                'so2': f"{so2:.2f}",
                'no2': f"{no2:.2f}",
            }
                
            row_data['pm25_verdict'] = get_pm25_verdict(pm25)
            row_data['pm10_verdict'] = get_pm10_verdict(pm10)
            row_data['co_verdict'] = get_co_verdict(co)
            row_data['so2_verdict'] = get_so2_verdict(so2)
            row_data['no2_verdict'] = get_no2_verdict(no2)
                
            day_pollutants = {
                'pm25': pm25, 'pm10': pm10, 'so2': so2, 'co': co, 'no2': no2,
            }
            verdict_sum = get_day_verdict(day_pollutants)
            if verdict_sum <= 4:
                row_data['day_verdict'] = 'Плохо'
            elif verdict_sum >= 8:
                row_data['day_verdict'] = 'Хорошо'
            else:
                row_data['day_verdict'] = 'Нормально'

            row_data = add_predictions(row_data, date_obj)
            data.append(row_data)
    
    with open(file_path1, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=",")
        for row in reader:
            date_str = row['date']
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                if date_obj.year == 2024:
                    date_key = date_str
                else:
                    continue
            except ValueError:
                continue
            
            if date_key in extra_data:
                co_val = extra_data[date_key]['co']
                no2_val = extra_data[date_key]['no2']
                so2_val = extra_data[date_key]['so2']
            else:
                co_val = 0
                no2_val = 0
                so2_val = 0
            
            row_data = {
                'date': f"{date_obj.day} {MONTHS_RU[date_obj.month - 1]} {date_obj.year}",
                'date_obj': date_obj,
                'pm25': f"{float(row.get('pm25', 0)):.2f}",
                'pm10': f"{float(row.get('pm10', 0)):.2f}",
                'co': f"{co_val:.2f}",
                'so2': f"{so2_val:.2f}",
                'no2': f"{no2_val:.2f}",
            }
            
            row_data['pm25_verdict'] = get_pm25_verdict(row_data['pm25'])
            row_data['pm10_verdict'] = get_pm10_verdict(row_data['pm10'])
            row_data['co_verdict'] = get_co_verdict(co_val)
            row_data['so2_verdict'] = get_so2_verdict(so2_val)
            row_data['no2_verdict'] = get_no2_verdict(no2_val)
            
            day_pollutants = {
                'pm25': float(row_data['pm25']), 'pm10': float(row_data['pm10']),
                'so2': so2_val, 'co': co_val, 'no2': no2_val,
            }
            verdict_sum = get_day_verdict(day_pollutants)
            if verdict_sum <= 4:
                row_data['day_verdict'] = 'Плохо'
            elif verdict_sum >= 8:
                row_data['day_verdict'] = 'Хорошо'
            else:
                row_data['day_verdict'] = 'Нормально'

            row_data = add_predictions(row_data, date_obj)
            data.append(row_data)
    
    data.sort(key=lambda x: x['date_obj'], reverse=True)
    
    return render(request, "pages/history.html", {
        "data": data,
    })
def about(request):
    return render(request, 'pages/about.html')