from django.shortcuts import render
import csv
import os
from django.conf import settings
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import numpy as np

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
        verdicts.append(get_o3_verdict(row.get('o3', '0')))
    except KeyError:
        return '—'

    for i in verdicts:
        if i == 'Нормально':
            total_sum += 1
        elif i == 'Хорошо':
            total_sum += 2

    return total_sum

def get_pm_forecast(forecast_rows):
    MODEL_PATH = Path(settings.BASE_DIR) / 'models' / 'pm1025_3day_forecast_model.pkl'
    model = joblib.load(MODEL_PATH)

    pm25_values = []
    pm10_values = []
    dates = []
    
    for row in forecast_rows:
        date_str = row[0].strip()
        date_obj = datetime.strptime(date_str, '%d.%m')
        date_obj = date_obj.replace(year=datetime.now().year)
        dates.append(date_obj)
        
        pm25 = round(float(row[1].strip().replace(',', '.')) * 1000, 2)
        pm10 = round(float(row[2].strip().replace(',', '.')) * 1000, 2)
        pm25_values.append(pm25)
        pm10_values.append(pm10)
    
    features = [
        dates[-1].timetuple().tm_yday,
        dates[-1].month,
        dates[-1].weekday(),
        pm25_values[-2],
        pm25_values[-3],
        pm25_values[-4],
        sum(pm25_values[-7:]) / 7,
        pm10_values[-2],
        pm10_values[-3],
        pm10_values[-4],
        sum(pm10_values[-7:]) / 7
    ]
    
    features_array = np.array([features])
    predictions = model.predict(features_array)[0]
    
    return {
        'pm25': {
            'day1': round(predictions[0], 1),
            'day2': round(predictions[1], 1),
            'day3': round(predictions[2], 1)
        },
        'pm10': {
            'day1': round(predictions[3], 1),
            'day2': round(predictions[4], 1),
            'day3': round(predictions[5], 1)
        }
    }

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
        'o3': 100,
    }
    aqi_sum = get_day_verdict(day_pollutants)
    
    if aqi_sum <= 4:
        quality_text = 'плохое'
        aqi_value = f'{aqi_sum}/12'
    else:
        if aqi_sum >= 9:
            quality_text = 'хорошее'
            aqi_value = f'{aqi_sum}/12'
        else:
            quality_text = 'нормальное'
            aqi_value = f'{aqi_sum}/12'

    forecast_rows = rows[-7:]

    pm_forecast = get_pm_forecast(forecast_rows) 

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
                'o3': {'value': '100', 'unit': 'мкг/м³', 'verdict': get_o3_verdict(100)},
            }
        },
        'predictions': [
            {
                'date': f'{tomorrow.day} {MONTHS_RU[tomorrow.month - 1]} {tomorrow.year}',
                'aqi': '7/10',
                'quality_verdict': 'хорошее',
                'pollutants': {
                    'pm25': {'value': f'{pm_forecast["pm25"]["day1"]}', 'unit': 'мкг/м³', 'verdict': get_pm25_verdict(pm_forecast['pm25']['day1'])},
                    'pm10': {'value': f'{pm_forecast["pm10"]["day1"]}', 'unit': 'мкг/м³', 'verdict': get_pm10_verdict(pm_forecast['pm10']['day1'])},
                    'so2': {'value': '35', 'unit': 'мкг/м³', 'verdict': 'Хорошо'},
                    'co': {'value': '3', 'unit': 'мг/м³', 'verdict': 'Хорошо'},
                    'no2': {'value': '20', 'unit': 'мкг/м³', 'verdict': 'Хорошо'},
                    'o3': {'value': '80', 'unit': 'мкг/м³', 'verdict': 'Хорошо'},
                }
            },
            {
                'date': f'{tomorrow1.day} {MONTHS_RU[tomorrow1.month - 1]} {tomorrow1.year}',
                'aqi': '5/10',
                'quality_verdict': 'нормальное',
                'pollutants': {
                    'pm25': {'value': f'{pm_forecast["pm25"]["day2"]}', 'unit': 'мкг/м³', 'verdict': get_pm25_verdict(pm_forecast['pm25']['day2'])},
                    'pm10': {'value': f'{pm_forecast["pm10"]["day2"]}', 'unit': 'мкг/м³', 'verdict': get_pm10_verdict(pm_forecast['pm10']['day2'])},
                    'so2': {'value': '80', 'unit': 'мкг/м³', 'verdict': 'Нормально'},
                    'co': {'value': '8', 'unit': 'мг/м³', 'verdict': 'Нормально'},
                    'no2': {'value': '100', 'unit': 'мкг/м³', 'verdict': 'Плохо'},
                    'o3': {'value': '160', 'unit': 'мкг/м³', 'verdict': 'Плохо'},
                }
            },
            {
                'date': f'{tomorrow2.day} {MONTHS_RU[tomorrow2.month - 1]} {tomorrow2.year}',
                'aqi': '6/10',
                'quality_verdict': 'нормальное',
                'pollutants': {
                    'pm25': {'value': f'{pm_forecast["pm25"]["day3"]}', 'unit': 'мкг/м³', 'verdict': get_pm25_verdict(pm_forecast['pm25']['day3'])},
                    'pm10': {'value': f'{pm_forecast["pm10"]["day3"]}', 'unit': 'мкг/м³', 'verdict': get_pm10_verdict(pm_forecast['pm10']['day3'])},
                    'so2': {'value': '60', 'unit': 'мкг/м³', 'verdict': 'Нормально'},
                    'co': {'value': '6', 'unit': 'мг/м³', 'verdict': 'Нормально'},
                    'no2': {'value': '50', 'unit': 'мкг/м³', 'verdict': 'Нормально'},
                    'o3': {'value': '130', 'unit': 'мкг/м³', 'verdict': 'Нормально'},
                }
            },
        ]
    }
    
    return render(request, 'pages/home.html', context)

def history(request):
    file_path1 = os.path.join(settings.BASE_DIR, "data", "ANALYSED-2024.csv")
    file_path2 = os.path.join(settings.BASE_DIR, "data", "CAMS-gases-2024.csv")
    file_path3 = Path(settings.BASE_DIR) / 'data' / 'pollutants.csv'
    
    extra_data = {}
    with open(file_path2, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            extra_data[row['date']] = {
                'co': round(float(row.get('CO', 0)), 2),
                'no2': round(float(row.get('NO2', 0)), 2),
                'o3': round(float(row.get('O3', 0)), 2),
                'so2': round(float(row.get('SO2', 0)), 2),
            }
    
    data = []
    
    if file_path3.exists():
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
                    'o3': '0.00',
                }
                
                row_data['pm25_verdict'] = get_pm25_verdict(pm25)
                row_data['pm10_verdict'] = get_pm10_verdict(pm10)
                row_data['co_verdict'] = get_co_verdict(co)
                row_data['so2_verdict'] = get_so2_verdict(so2)
                row_data['no2_verdict'] = get_no2_verdict(no2)
                row_data['o3_verdict'] = get_o3_verdict(0)
                
                day_pollutants = {
                    'pm25': pm25,
                    'pm10': pm10,
                    'so2': so2,
                    'co': co,
                    'no2': no2,
                    'o3': 0,
                }
                verdict_sum = get_day_verdict(day_pollutants)
                if verdict_sum <= 4:
                    row_data['day_verdict'] = 'Плохо'
                elif verdict_sum >= 9:
                    row_data['day_verdict'] = 'Хорошо'
                else:
                    row_data['day_verdict'] = 'Нормально'
                
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
                o3_val = extra_data[date_key]['o3']
                so2_val = extra_data[date_key]['so2']
            else:
                co_val = 0
                no2_val = 0
                o3_val = 0
                so2_val = 0
            
            row_data = {
                'date': f"{date_obj.day} {MONTHS_RU[date_obj.month - 1]} {date_obj.year}",
                'date_obj': date_obj,
                'pm25': f"{float(row.get('pm25', 0)):.2f}",
                'pm10': f"{float(row.get('pm10', 0)):.2f}",
                'co': f"{co_val:.2f}",
                'so2': f"{so2_val:.2f}",
                'no2': f"{no2_val:.2f}",
                'o3': f"{o3_val:.2f}",
            }
            
            row_data['pm25_verdict'] = get_pm25_verdict(row_data['pm25'])
            row_data['pm10_verdict'] = get_pm10_verdict(row_data['pm10'])
            row_data['co_verdict'] = get_co_verdict(co_val)
            row_data['so2_verdict'] = get_so2_verdict(so2_val)
            row_data['no2_verdict'] = get_no2_verdict(no2_val)
            row_data['o3_verdict'] = get_o3_verdict(o3_val)
            
            day_pollutants = {
                'pm25': float(row_data['pm25']),
                'pm10': float(row_data['pm10']),
                'so2': so2_val,
                'co': co_val,
                'no2': no2_val,
                'o3': o3_val,
            }
            verdict_sum = get_day_verdict(day_pollutants)
            if verdict_sum <= 4:
                row_data['day_verdict'] = 'Плохо'
            elif verdict_sum >= 9:
                row_data['day_verdict'] = 'Хорошо'
            else:
                row_data['day_verdict'] = 'Нормально'
            
            data.append(row_data)
    
    data.sort(key=lambda x: x['date_obj'], reverse=True)
    
    return render(request, "pages/history.html", {
        "data": data,
    })
def about(request):
    return render(request, 'pages/about.html')