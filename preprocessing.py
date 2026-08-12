import pandas as pd
import holidays
import numpy as np

def preprocess_sales_data(df):
    # 날짜 데이터 타입 변환
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df.rename(columns = {
    'Date' : '날짜', 
    'Sales' : '일매출', 
    'Labor_Cost' : '인건비용', 
    'Material_Cost' : '식자재비용', 
    'Other_Expense' : '기타비용', 
    'Table_Count' : '테이블수'}, 
    inplace = True)
    
    # 시계열 순서 보장을 위한 정렬 (필수)
    df = df.sort_values('날짜').reset_index(drop=True)

    df['월'] = df['날짜'].dt.month
    df['요일'] = df['날짜'].dt.dayofweek # 0:월, 6:일 추가
    df['일'] = df['날짜'].dt.day

    # 시간의 주기성 반영 (12월-1월, 31일-1일의 연속성 학습)
    df['월_sin'] = np.sin(2 * np.pi * df['월'] / 12)
    df['월_cos'] = np.cos(2 * np.pi * df['월'] / 12)
    df['일_sin'] = np.sin(2 * np.pi * df['일'] / 31)
    df['일_cos'] = np.cos(2 * np.pi * df['일'] / 31)
    #  요일 숫자를 문자열 이름으로 매핑 (더미 변수 컬럼명을 이쁘게 만들기 위함)
    weekday_map = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
    df['요일'] = df['요일'].map(weekday_map)
    # 요일 데이터 더미화 (One-Hot Encoding)
    # 다중공선성(Multicollinearity) 방지를 위해 drop_first=True를 쓸 수도 있으나, 
    # 트리 기반 모델(XGBoost)은 그냥 모두 쓰는 것이 직관적이고 성능에 유리함.
    df_dummies = pd.get_dummies(df['요일'], prefix='요일', dtype=int)
    df = pd.concat([df, df_dummies], axis=1)

    # 시계열 피처 생성
    df['전일매출'] = df['일매출'].shift(1)
    df['7일평균매출'] = df['일매출'].shift(1).rolling(window=7).mean()
    df['연월'] = df['날짜'].dt.strftime('%Y%m').astype(int)
    df['전주매출'] = df['일매출'].shift(7)
    df['14일평균매출'] = df['일매출'].shift(1).rolling(window=14).mean()
    df['28일평균매출'] = df['일매출'].shift(1).rolling(window=28).mean()
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)

    # 연말특수지수 파생변수 생성
    df['연말특수지수'] = 0
    # 1. 12월 조건 정의
    is_december = df['날짜'].dt.month == 12
    # 2. 요일 조건 정의 (평일 목금은 회사 회식, 주말 토요일은 동호회, 모임 회식)
    is_weekend_rush = (df['요일_목'] == 1) | (df['요일_금'] == 1) | (df['요일_토'] == 1)
    # 3. 연말 피크 시즌 조건 정의 (12월 22일 ~ 31일)
    is_end_of_year = df['날짜'].dt.day >= 22
    # 4. 조건 경합하여 연말특수지수를 1로 채우기
    # 조건 A: 12월이면서 평일
    df.loc[is_december & is_weekend_rush, '연말특수지수'] = 1
    
    # 조건 B: 12월이면서 22일~31일 (대관 및 연말 예약 폭주 기간)
    df.loc[is_december & is_end_of_year, '연말특수지수'] = 1

    return df

def preprocess_weather_data(weather_data):
    weather_df = weather_data.rename(columns={
        '일시': '날짜', 
        '평균기온(°C)': '평균기온',
        '강수 계속시간(hr)' : '강수계속시간'})
    weather_df = weather_df.drop(['지점', '지점명'], axis=1)
    weather_df['날짜'] = pd.to_datetime(weather_df['날짜'], format='%Y-%m-%d')
    # 비올확률 파생변수 (노트북 로직)
    weather_df['비올확률'] = (weather_df['강수계속시간'] > 0).astype(int)
    weather_df = weather_df.fillna(0)
    return weather_df

def preprocess_cpi_data(cpi_data):
    cpi_data['계정항목'] = cpi_data['계정항목'].str.strip()
    month_cols = ['Feb-25', 'Mar-25', 'Apr-25', 'May-25', 'Jun-25', 'Jul-25', 'Aug-25', 'Sep-25', 'Oct-25', 'Nov-25', 'Dec-25', 'Jan-26']

    def get_weighted_avg(row_indices):
        subset = cpi_data.iloc[row_indices]
        weights = subset['가중치']
        weighted_avg = {}
        for month in month_cols:
            weighted_sum = (subset[month] * weights).sum()
            total_weight = weights.sum()
            weighted_avg[month] = weighted_sum / total_weight
        return pd.Series(weighted_avg)

    # 전체 품목 및 식료품 가중 평균 산출
    # (수정 이력) get_weighted_avg(2)처럼 정수를 넘기면 cpi_data.iloc[2]가 DataFrame이 아니라
    # 단일 행 Series를 반환해, 가중평균 계산이 (값*가중치)/가중치로 셀프캔슬되어 사실상
    # '육류' 항목의 원본 물가지수를 그대로 반환하는 것과 같았음(결과값 자체는 우연히 같았지만
    # 계산 로직은 의도와 다르게 동작). 리스트로 감싸 명시적으로 DataFrame을 넘기도록 수정.
    meat_weighted_avg = get_weighted_avg([2])
    food_weighted_avg = get_weighted_avg(range(5))

    df_cpi_result = pd.DataFrame({
        '육류가중평균': meat_weighted_avg,
        '식료품가중평균': food_weighted_avg
    })
    
    # 연월 형식 변환
    month_map = {
        'Feb-25': 202502, 'Mar-25': 202503, 'Apr-25': 202504, 'May-25': 202505,
        'Jun-25': 202506, 'Jul-25': 202507, 'Aug-25': 202508, 'Sep-25': 202509,
        'Oct-25': 202510, 'Nov-25': 202511, 'Dec-25': 202512, 'Jan-26': 202601
    }
    df_cpi_result.rename(index=month_map, inplace=True)
    df_cpi_result = df_cpi_result.reset_index()
    df_cpi_result.rename(columns = {'index' : '연월'}, inplace = True)
    return df_cpi_result

def add_holiday_features(holiday_df, year):
    kr_holidays = holidays.KR(years = year)
    holiday_list = [d.strftime('%Y-%m-%d') for d in kr_holidays]
    # (수정 이력) '날짜'(datetime64) 대 holiday_list(str) 비교라 dtype이 불일치함.
    # pandas 2.x는 현재 암묵적 캐스팅으로 매칭해주지만 FutureWarning과 함께 향후 버전에서
    # 제거될 동작이라, 문자열로 명시 변환한 뒤 비교하도록 안전장치를 추가함.
    holiday_df['공휴일여부'] = holiday_df['날짜'].dt.strftime('%Y-%m-%d').isin(holiday_list).astype(int)
    holiday_df['공휴일전날'] = holiday_df['공휴일여부'].shift(-1).fillna(0).astype(int)
    
    # 휴일지수: 금(4), 토(5), 일(6) + 공휴일 + 공휴일전날
    # 요일 컬럼은 위에서 생성하므로 여기서는 계산용으로만 활용
    holiday_df['휴일지수'] = ((holiday_df['요일'].isin(['금', '토', '일'])) | (holiday_df['공휴일여부'] | holiday_df['공휴일전날'])).astype(int)
    holiday_df.drop(['요일','공휴일여부', '공휴일전날', '월'], axis=1, inplace=True)
    return holiday_df

def merge_data(sales_data, weather_data, cpi_data):
    sales_df = preprocess_sales_data(sales_data)
    weather_df = preprocess_weather_data(weather_data)
    cpi_df = preprocess_cpi_data(cpi_data)
    df = pd.merge(sales_df, weather_df, on='날짜', how='inner')
    df = pd.merge(df, cpi_df, on='연월', how='left')
    df = add_holiday_features(df, [2025, 2026])
    
    # 최장 이동평균인 28일치 결측치 제거
    df = df.dropna(subset=['28일평균매출', '전일매출']).reset_index(drop=True)
    df = df.fillna(0) # 기타 외부 변수 결측치만 0 처리
    df.drop(columns=['연월'], inplace=True)
    return df