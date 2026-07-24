import pandas as pd
import os
from datetime import datetime, timedelta
import config

# 원본 엑셀 서식(고정 양식) 상의 셀 위치 (0-indexed row, col)
# 표기는 엑셀 셀 주소(A1 표기법) 기준, 노트북 원본 주석을 그대로 이전함
CELL_DAILY_SALES = (24, 0)     # A25: 일매출
CELL_LABOR_COST = (26, 8)      # I27: 인건비용
CELL_MATERIAL_COST = (22, 8)   # I23: 식자재비용
CELL_OTHER_EXPENSE = (22, 10)  # K23: 기타경비비용
CELL_TABLE_COUNT = (19, 5)     # F20: 테이블수

def data_load():
    file_list = sorted([f for f in os.listdir(config.RAW_DATA_DIR) if f.endswith('.xlsx')]) # 폴더 내 엑셀 파일 목록

    all_data = []

    current_date = datetime(2025, 2, 1)

    for file_name in file_list:
        file_full_path = os.path.join(config.RAW_DATA_DIR, file_name)
        

        all_sheets = pd.read_excel(file_full_path, sheet_name=None, header=None) # 모든 시트 불러오기
        sheet_items = list(all_sheets.items())[2:]  # 두 번째 시트부터 추출
        
        for sheet_name, df in sheet_items:

            try:
                # 시트가 최소 요구 행/열 수보다 작은지 체크 (IndexError 방지)
                if df.shape[0] < 27 or df.shape[1] < 11:
                    continue
                    
                daily_sales = df.iloc[CELL_DAILY_SALES]
                labor_cost = df.iloc[CELL_LABOR_COST]
                material_cost = df.iloc[CELL_MATERIAL_COST]
                other_expense = df.iloc[CELL_OTHER_EXPENSE]
                table_count = df.iloc[CELL_TABLE_COUNT]

                all_data.append({
                    'Date': current_date.strftime('%Y%m%d'),
                    'Sales': daily_sales,
                    'Labor_Cost': labor_cost,
                    'Material_Cost': material_cost,
                    'Other_Expense': other_expense,
                    'Table_Count': table_count
                })
                current_date += timedelta(days=1)
            except Exception as e:
                print(f"오류 발생 - 파일: {file_name}, 시트: {sheet_name}, 내용: {e}")

    # 최종 데이터프레임 생성 및 저장
    df = pd.DataFrame(all_data)
    cols_to_fix = ['Sales', 'Labor_Cost', 'Material_Cost', 'Other_Expense', 'Table_Count']
    df[cols_to_fix] = df[cols_to_fix].apply(pd.to_numeric, errors='coerce').fillna(0)

    return df

def load_weather_data():
    weather_data = pd.read_csv(config.WEATHER_DATA_PATH, encoding = 'cp949')
    return weather_data

def load_cpi_data():
    cpi_data = pd.read_csv(config.CPI_DATA_PATH, encoding = 'cp949')
    return cpi_data