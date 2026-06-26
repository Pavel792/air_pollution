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

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PM25_MODEL = joblib.load(MODELS_DIR / "model_pm25.pkl")


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
    for n in range(1, 8):
        df[f"{n}DayAgo"] = df[target].shift(n)
    df["3DaysMean"] = df[target].rolling(window=3, min_periods=1).mean()
    df["7DaysMean"] = df[target].rolling(window=7, min_periods=1).mean()
    df["3DaysMax"]  = df[target].rolling(window=3, min_periods=1).max()
    df["3DaysMin"]  = df[target].rolling(window=3, min_periods=1).min()
    df["7DaysStd"]  = df[target].rolling(window=7, min_periods=1).std()
    df["trend"]     = df[target] - df[target].shift(1)
    return df


def get_forecast_lr(target, data):
    features = [
        "dayNumber", "monthNumber", "dayOfWeek", "isWeekend",
        "1DayAgo", "2DayAgo", "3DayAgo", "4DayAgo",
        "5DayAgo", "6DayAgo", "7DayAgo",
        "3DaysMean", "7DaysMean", "3DaysMax", "3DaysMin",
        "7DaysStd", "trend"
    ]

    transformed = transform_data(data, target).dropna()

    if len(transformed) < 3:
        return {"day1": 0, "day2": 0, "day3": 0}

    model = LinearRegression()
    model.fit(transformed[features], transformed[target])

    results = []
    last_values = data[target].values.tolist()
    last_date = data["date"].iloc[-1]

    for day in range(3):
        next_date    = last_date + pd.Timedelta(days=1)
        day_number   = next_date.timetuple().tm_yday
        month_number = next_date.month
        day_of_week  = next_date.weekday()
        is_weekend   = 1 if day_of_week >= 5 else 0
        last_7       = last_values[-7:] if len(last_values) >= 7 else last_values

        X_pred = np.array([[
            day_number, month_number, day_of_week, is_weekend,
            last_values[-1],
            last_values[-2] if len(last_values) >= 2 else last_values[-1],
            last_values[-3] if len(last_values) >= 3 else last_values[-1],
            last_values[-4] if len(last_values) >= 4 else last_values[-1],
            last_values[-5] if len(last_values) >= 5 else last_values[-1],
            last_values[-6] if len(last_values) >= 6 else last_values[-1],
            last_values[-7] if len(last_values) >= 7 else last_values[-1],
            sum(last_values[-3:]) / min(3, len(last_values)),
            sum(last_7) / len(last_7),
            max(last_values[-3:]),
            min(last_values[-3:]),
            np.std(last_7) if len(last_7) >= 2 else 0,
            last_values[-1] - last_values[-2] if len(last_values) >= 2 else 0,
        ]])

        pred = max(0.0, model.predict(X_pred)[0])
        results.append(round(pred, 2))
        last_values.append(pred)
        last_date = next_date

    return {"day1": results[0], "day2": results[1], "day3": results[2]}


def get_forecast_pm25(data):
    last_values = data["pm25"].values.tolist()
    results = []

    for day in range(3):
        X_pred = pd.DataFrame([{
            "1DayAgo":   last_values[-1],
            "2DayAgo":   last_values[-2],
            "3DayAgo":   last_values[-3],
            "7DaysMean": sum(last_values[-7:]) / min(7, len(last_values)),
        }])
        pred = max(0.0, PM25_MODEL.predict(X_pred)[0])
        results.append(round(pred, 2))
        last_values.append(pred)

    return {"day1": results[0], "day2": results[1], "day3": results[2]}


def load_pollutants():
    file_path = os.path.join(settings.BASE_DIR, "data", "pollutants.csv")
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=";")
        rows = list(reader)

    col_ind = {"pm25": 1, "pm10": 2, "co": 3, "so2": 4, "no2": 5}
    result = {}

    for target in ["pm25", "pm10", "co", "so2", "no2"]:
        data = [[row[0], row[col_ind[target]]] for row in rows[-21:]]
        df = pd.DataFrame(data, columns=["date", target])
        df["date"]   = pd.to_datetime(df["date"].astype(str).str.zfill(5), format="%d.%m")
        df[target]   = df[target].str.replace(",", ".").astype(float)
        if target in ["pm25", "pm10", "so2", "no2"]:
            df[target] = df[target] * 1000
        df["dayNumber"]   = df["date"].dt.dayofyear
        df["monthNumber"] = df["date"].dt.month
        df["dayOfWeek"]   = df["date"].dt.dayofweek
        df["isWeekend"]   = (df["dayOfWeek"] >= 5).astype(int)
        result[target] = df

    return result, rows[-1]


def get_forcast_verdicts(pm25_forecast, pm10_forecast, so2_forecast, co_forecast, no2_forecast):
    def aqi(forecasts):
        return sum([2 if i == 'Хорошо' else 1 if i == 'Нормально' else 0 for i in forecasts])

    def verdict(score):
        return "плохое" if score <= 4 else "хорошее" if score >= 8 else "нормальное"

    day1_list = [get_pm25_verdict(pm25_forecast["day1"]), get_pm10_verdict(pm10_forecast["day1"]), get_so2_verdict(so2_forecast["day1"]), get_co_verdict(co_forecast["day1"]), get_no2_verdict(no2_forecast["day1"])]
    day2_list = [get_pm25_verdict(pm25_forecast["day2"]), get_pm10_verdict(pm10_forecast["day2"]), get_so2_verdict(so2_forecast["day2"]), get_co_verdict(co_forecast["day2"]), get_no2_verdict(no2_forecast["day2"])]
    day3_list = [get_pm25_verdict(pm25_forecast["day3"]), get_pm10_verdict(pm10_forecast["day3"]), get_so2_verdict(so2_forecast["day3"]), get_co_verdict(co_forecast["day3"]), get_no2_verdict(no2_forecast["day3"])]

    day1_aqi = aqi(day1_list)
    day2_aqi = aqi(day2_list)
    day3_aqi = aqi(day3_list)

    return {
        "day1_aqi": day1_aqi, "day1_verdict": verdict(day1_aqi),
        "day2_aqi": day2_aqi, "day2_verdict": verdict(day2_aqi),
        "day3_aqi": day3_aqi, "day3_verdict": verdict(day3_aqi),
    }


def home(request):
    today     = datetime.now()
    tomorrow  = datetime.now() + timedelta(days=1)
    tomorrow1 = datetime.now() + timedelta(days=2)
    tomorrow2 = datetime.now() + timedelta(days=3)

    data_by_target, today_row = load_pollutants()

    pm25_float = round(float(today_row[1].strip().replace(',', '.')) * 1000, 2)
    pm10_float = round(float(today_row[2].strip().replace(',', '.')) * 1000, 2)
    CO_float   = round(float(today_row[3].strip().replace(',', '.')), 2)
    SO2_float  = round(float(today_row[4].strip().replace(',', '.')) * 1000, 2)
    NO2_float  = round(float(today_row[5].strip().replace(',', '.')) * 1000, 2)

    day_pollutants = {'pm25': pm25_float, 'pm10': pm10_float, 'so2': SO2_float, 'co': CO_float, 'no2': NO2_float}
    aqi_sum      = get_day_verdict(day_pollutants)
    quality_text = 'плохое' if aqi_sum <= 4 else 'хорошее' if aqi_sum >= 8 else 'нормальное'
    aqi_value    = f'{aqi_sum}/10'

    pm25_forecast = get_forecast_pm25(data_by_target["pm25"])
    pm10_forecast = get_forecast_lr("pm10", data_by_target["pm10"])
    so2_forecast  = get_forecast_lr("so2",  data_by_target["so2"])
    co_forecast   = get_forecast_lr("co",   data_by_target["co"])
    no2_forecast  = get_forecast_lr("no2",  data_by_target["no2"])

    forecast_verdicts = get_forcast_verdicts(pm25_forecast, pm10_forecast, so2_forecast, co_forecast, no2_forecast)

    def make_day(date, day_key, fv):
        return {
            'date': f'{date.day} {MONTHS_RU[date.month - 1]} {date.year}',
            'aqi': f'{fv[f"{day_key}_aqi"]}/10',
            'quality_verdict': fv[f"{day_key}_verdict"],
            'pollutants': {
                'pm25': {'value': f'{pm25_forecast[day_key]}', 'unit': 'мкг/м³', 'verdict': get_pm25_verdict(pm25_forecast[day_key])},
                'pm10': {'value': f'{pm10_forecast[day_key]}', 'unit': 'мкг/м³', 'verdict': get_pm10_verdict(pm10_forecast[day_key])},
                'so2':  {'value': f'{so2_forecast[day_key]}',  'unit': 'мкг/м³', 'verdict': get_so2_verdict(so2_forecast[day_key])},
                'co':   {'value': f'{co_forecast[day_key]}',   'unit': 'мкг/м³', 'verdict': get_co_verdict(co_forecast[day_key])},
                'no2':  {'value': f'{no2_forecast[day_key]}',  'unit': 'мкг/м³', 'verdict': get_no2_verdict(no2_forecast[day_key])},
            }
        }

    context = {
        'today': {
            'date': f'{today.day} {MONTHS_RU[today.month - 1]} {today.year}',
            'aqi': aqi_value,
            'quality_verdict': quality_text,
            'pollutants': {
                'pm25': {'value': f'{pm25_float}', 'unit': 'мкг/м³', 'verdict': get_pm25_verdict(pm25_float)},
                'pm10': {'value': f'{pm10_float}', 'unit': 'мкг/м³', 'verdict': get_pm10_verdict(pm10_float)},
                'so2':  {'value': f'{SO2_float}',  'unit': 'мкг/м³', 'verdict': get_so2_verdict(SO2_float)},
                'co':   {'value': f'{CO_float}',   'unit': 'мг/м³',  'verdict': get_co_verdict(CO_float)},
                'no2':  {'value': f'{NO2_float}',  'unit': 'мкг/м³', 'verdict': get_no2_verdict(NO2_float)},
            }
        },
        'predictions': [
            make_day(tomorrow,  "day1", forecast_verdicts),
            make_day(tomorrow1, "day2", forecast_verdicts),
            make_day(tomorrow2, "day3", forecast_verdicts),
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
                'co':  round(float(row.get('CO', 0)), 2),
                'no2': round(float(row.get('NO2', 0)), 2),
                'so2': round(float(row.get('SO2', 0)), 2),
            }

    predictions_data = {}
    with open(file_path4, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            date_str = row['date'].strip().zfill(5)
            try:
                date_obj = datetime.strptime(date_str, '%d.%m').replace(year=datetime.now().year)
            except ValueError:
                continue
            predictions_data[date_obj.strftime('%Y-%m-%d')] = {
                'pm25': round(float(row.get('pm25', 0)), 2),
                'pm10': round(float(row.get('pm10', 0)), 2),
                'co':   round(float(row.get('CO', 0)), 2),
                'so2':  round(float(row.get('SO2', 0)), 2),
                'no2':  round(float(row.get('NO2', 0)), 2),
            }

    def add_predictions(row_data, date_obj):
        pred = predictions_data.get(date_obj.strftime('%Y-%m-%d'), {})
        row_data['pred_pm25'] = f"{pred.get('pm25', 0):.2f}"
        row_data['pred_pm10'] = f"{pred.get('pm10', 0):.2f}"
        row_data['pred_co']   = f"{pred.get('co', 0):.2f}"
        row_data['pred_so2']  = f"{pred.get('so2', 0):.2f}"
        row_data['pred_no2']  = f"{pred.get('no2', 0):.2f}"
        row_data['pred_pm25_verdict'] = get_pm25_verdict(pred.get('pm25', 0))
        row_data['pred_pm10_verdict'] = get_pm10_verdict(pred.get('pm10', 0))
        row_data['pred_co_verdict']   = get_co_verdict(pred.get('co', 0))
        row_data['pred_so2_verdict']  = get_so2_verdict(pred.get('so2', 0))
        row_data['pred_no2_verdict']  = get_no2_verdict(pred.get('no2', 0))
        return row_data

    def day_verdict_text(verdict_sum):
        return 'Плохо' if verdict_sum <= 4 else 'Хорошо' if verdict_sum >= 8 else 'Нормально'

    today_date = datetime.now().date()
    data = []

    with open(file_path3, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                date_obj = datetime.strptime(row[0].strip().zfill(5), '%d.%m').replace(year=datetime.now().year)
            except ValueError:
                continue
            if date_obj.date() > today_date:
                continue

            pm25 = round(float(row[1].strip().replace(',', '.')) * 1000, 2)
            pm10 = round(float(row[2].strip().replace(',', '.')) * 1000, 2)
            co   = round(float(row[3].strip().replace(',', '.')), 2)
            so2  = round(float(row[4].strip().replace(',', '.')) * 1000, 2)
            no2  = round(float(row[5].strip().replace(',', '.')) * 1000, 2)

            row_data = {
                'date': f"{date_obj.day} {MONTHS_RU[date_obj.month - 1]} {date_obj.year}",
                'date_obj': date_obj,
                'pm25': f"{pm25:.2f}", 'pm10': f"{pm10:.2f}",
                'co': f"{co:.2f}", 'so2': f"{so2:.2f}", 'no2': f"{no2:.2f}",
                'pm25_verdict': get_pm25_verdict(pm25),
                'pm10_verdict': get_pm10_verdict(pm10),
                'co_verdict':   get_co_verdict(co),
                'so2_verdict':  get_so2_verdict(so2),
                'no2_verdict':  get_no2_verdict(no2),
                'day_verdict':  day_verdict_text(get_day_verdict({'pm25': pm25, 'pm10': pm10, 'so2': so2, 'co': co, 'no2': no2})),
            }
            row_data = add_predictions(row_data, date_obj)
            data.append(row_data)

    with open(file_path1, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=",")
        for row in reader:
            try:
                date_obj = datetime.strptime(row['date'], '%Y-%m-%d')
                if date_obj.year != 2024:
                    continue
            except ValueError:
                continue

            date_key = row['date']
            co_val  = extra_data.get(date_key, {}).get('co', 0)
            no2_val = extra_data.get(date_key, {}).get('no2', 0)
            so2_val = extra_data.get(date_key, {}).get('so2', 0)
            pm25_val = float(row.get('pm25', 0))
            pm10_val = float(row.get('pm10', 0))

            row_data = {
                'date': f"{date_obj.day} {MONTHS_RU[date_obj.month - 1]} {date_obj.year}",
                'date_obj': date_obj,
                'pm25': f"{pm25_val:.2f}", 'pm10': f"{pm10_val:.2f}",
                'co':  f"{co_val:.2f}", 'so2': f"{so2_val:.2f}", 'no2': f"{no2_val:.2f}",
                'pm25_verdict': get_pm25_verdict(pm25_val),
                'pm10_verdict': get_pm10_verdict(pm10_val),
                'co_verdict':   get_co_verdict(co_val),
                'so2_verdict':  get_so2_verdict(so2_val),
                'no2_verdict':  get_no2_verdict(no2_val),
                'day_verdict':  day_verdict_text(get_day_verdict({'pm25': pm25_val, 'pm10': pm10_val, 'so2': so2_val, 'co': co_val, 'no2': no2_val})),
            }
            row_data = add_predictions(row_data, date_obj)
            data.append(row_data)

    data.sort(key=lambda x: x['date_obj'], reverse=True)
    return render(request, "pages/history.html", {"data": data})


def about(request):
    return render(request, 'pages/about.html')