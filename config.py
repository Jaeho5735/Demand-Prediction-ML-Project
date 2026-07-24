from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_DIR = BASE_DIR / 'xlsx'
WEATHER_DATA_PATH = BASE_DIR / 'data' / 'seoul_weather.csv'
CPI_DATA_PATH = BASE_DIR / 'data' / 'CPI.csv'
