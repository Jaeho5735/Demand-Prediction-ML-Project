import sys
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import data_load, load_weather_data, load_cpi_data
from preprocessing import merge_data
from model import train_and_evaluate, plot_results # 시각화 함수 임포트

def main():
    # 터미널 한글 깨짐 방지 (Windows 환경)
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    print("1. 데이터 소스로부터 파일들을 로드하는 중...")
    raw_sales = data_load()
    raw_weather = load_weather_data()
    raw_cpi = load_cpi_data()
    
    print("2. 데이터를 전처리하는 중...")
    # merge_data 내부에서 나머지 전처리 함수들이 호출됨
    df = merge_data(raw_sales, raw_weather, raw_cpi)

    # 피처 리스트
    features = ['인건비용', '식자재비용', '기타비용', '테이블수', '평균기온', '강수계속시간',
             '전체품목가중평균', '식료품가중평균', '휴일지수', '월', '비올확률', '전일매출', '7일평균매출']
    
    X = df[features]
    y = df['일매출']

    # 과거 데이터로 미래를 예측하므로 셔플 없이 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("3. 모델 학습 및 평가를 진행하는 중...")
    model, pred = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("4. 예측 결과 시각화 중...")
    plot_results(model, y_test, pred, features)

    print("프로세스 완료")

if __name__ == "__main__":
    main()