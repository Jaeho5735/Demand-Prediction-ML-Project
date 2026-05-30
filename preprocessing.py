import pandas as pd
import holidays

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
    
    # 비용 반올림 처리
    df['인건비용'] = df['인건비용'].round(-2)
    df['기타비용'] = df['기타비용'].round(-2)

    df['월'] = df['날짜'].dt.month
    # 시계열 피처 생성
    df['전일매출'] = df['일매출'].shift(1)
    df['7일평균매출'] = df['일매출'].rolling(window=7).mean()
    df['연월'] = df['날짜'].dt.strftime('%Y%m').astype(int)

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
    total_weighted_avg = get_weighted_avg(cpi_data.index)
    food_weighted_avg = get_weighted_avg(range(5))

    df_cpi_result = pd.DataFrame({
        '전체품목가중평균': total_weighted_avg,
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
    holiday_df['요일'] = holiday_df['날짜'].dt.dayofweek
    holiday_df['공휴일여부'] = holiday_df['날짜'].isin(holiday_list).astype(int)
    holiday_df['공휴일전날'] = holiday_df['공휴일여부'].shift(-1).fillna(0).astype(int)
    
    # 휴일지수: 금(4), 토(5), 일(6) + 공휴일 + 공휴일전날
    holiday_df['휴일지수'] = ((holiday_df['요일'].isin([4, 5, 6])) | (holiday_df['공휴일여부'] | holiday_df['공휴일전날'])).astype(int)
    holiday_df.drop(['요일', '공휴일여부', '공휴일전날'],axis=1, inplace=True)
    return holiday_df

def merge_data(sales_data, weather_data, cpi_data):
    sales_df = preprocess_sales_data(sales_data)
    weather_df = preprocess_weather_data(weather_data)
    cpi_df = preprocess_cpi_data(cpi_data)
    df = pd.merge(sales_df, weather_df, on='날짜', how='inner')
    df = pd.merge(df, cpi_df, on='연월', how='left')
    df = add_holiday_features(df, [2025, 2026])
    
    # 결측치(이동평균 등) 제거 및 인덱스 정리
    df = df.fillna(0).iloc[7:].reset_index(drop=True)
    df.drop(columns=['연월'], inplace=True)
    return df