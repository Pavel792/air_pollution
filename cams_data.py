import cdsapi
import xarray as xr
from datetime import datetime, timedelta
import os

def download_cams_nrt(start_date, end_date, output_file):
    """
    Скачивает актуальные данные CAMS NRT за период
    Доступно: с 2016 по текущую дату (задержка ~3-5 дней)
    """
    
    client = cdsapi.Client()
    
    # Для NRT используем forecast датасет с type='forecast'
    request = {
        'date': f'{start_date}/{end_date}',
        'time': '00:00',
        'leadtime_hour': '0',           # анализ на момент времени
        'type': 'forecast',              # NRT прогноз
        'variable': [
            'particulate_matter_2.5um',
            'particulate_matter_10um',
            'nitrogen_dioxide',
            'sulphur_dioxide',
            'carbon_monoxide',
            'ozone',
        ],
        'area': [60.2, 29.8, 59.7, 30.8],  # Петербург
        'format': 'netcdf',
    }
    
    print(f"Скачиваю NRT данные с {start_date} по {end_date}...")
    print("Доступный период: 2016-2026 (с задержкой 3-5 дней)")
    
    client.retrieve('cams-global-atmospheric-composition-forecasts', request, output_file)
    print(f"Скачано! Файл: {output_file}")
    
    return output_file

def get_recent_data():
    """
    Стягивает данные за последнюю доступную неделю
    """
    # Берем дату 5 дней назад (учитываем задержку NRT)
    end_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=12)).strftime('%Y-%m-%d')
    
    output_file = 'cams_nrt_last_week.nc'
    download_cams_nrt(start_date, end_date, output_file)
    
    # Парсим в словарь
    ds = xr.open_dataset(output_file)
    result = {}
    
    for i, time in enumerate(ds.time.values):
        date_str = str(time)[:10]
        result[date_str] = {}
        
        try:
            result[date_str]['pm25'] = float(ds.pm2p5.isel(time=i).mean().values)
            result[date_str]['pm10'] = float(ds.pm10.isel(time=i).mean().values)
            result[date_str]['no2'] = float(ds.no2.isel(time=i).mean().values)
            result[date_str]['so2'] = float(ds.so2.isel(time=i).mean().values)
            result[date_str]['co'] = float(ds.co.isel(time=i).mean().values)
            result[date_str]['o3'] = float(ds.o3.isel(time=i).mean().values)
        except Exception as e:
            print(f"Ошибка для {date_str}: {e}")
            continue
    
    return result

# === ИСПОЛЬЗОВАНИЕ ===
if __name__ == '__main__':
    data = get_recent_data()
    print("\nЗагруженные данные:")
    for date, values in data.items():
        print(f"{date}: PM2.5={values['pm25']:.1f} мкг/м³")