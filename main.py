import sys
from sklearn.model_selection import train_test_split
from data_loader import data_load, load_weather_data, load_cpi_data
from preprocessing import merge_data
from model import train_and_evaluate, plot_results, walk_forward_evaluate, evaluate_baselines

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
    # (수정 이력) '인건비용', '식자재비용', '기타비용', '테이블수'는 모두 그날 영업이 마감되어야
    # 확정되는 값이라 예측 시점(전날/당일 아침)에는 알 수 없는 정보임 (운영 가능성 위반 -> 3번째 유형의 데이터 누수).
    # 특히 '테이블수'는 그날 손님 수와 사실상 동일해 매출을 매출로 맞히는 것과 다름없어 전부 제외함.
    features = ['평균기온', '강수계속시간',
             '육류가중평균', '식료품가중평균', '휴일지수', '비올확률', '전일매출', '7일평균매출', '요일_월', '요일_화',
            '요일_수', '요일_목', '요일_금', '요일_토', '요일_일', '전주매출', '14일평균매출', '28일평균매출', '연말특수지수', '월_sin', '월_cos', '일_sin', '일_cos']
    
    X = df[features]
    y = df['일매출']

    # 시계열 데이터이므로 시간 순서를 보존한 채 마지막 20%를 테스트셋으로 고정
    # (shuffle=True를 쓰면 테스트 시점의 인접 날짜가 학습셋에 섞여 들어가 미래 정보 누수가 발생함)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("3. 모델 학습 및 평가를 진행하는 중...")
    model, pred = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("4. 베이스라인 대비 성능 비교 중...")
    # XGBoost 모델이 "아무 규칙 없이 어제 매출/7일 평균을 그대로 예측한 것"보다
    # 실제로 더 나은지 확인 (R2/MAE 단독 수치만으로는 모델의 가치를 판단할 수 없기 때문)
    evaluate_baselines(X_test, y_test)

    print("5. 예측 결과 시각화 중...")
    plot_results(model, y_test, pred, features)

    print("6. Walk-Forward 다중 폴드 검증 진행 중...")
    # 마지막 20% 단일 홀드아웃이 특정 시즌에 쏠리는 문제를 보완하기 위해
    # 동일 하이퍼파라미터로 여러 시점을 기준으로 재검증 (베이스라인도 폴드마다 함께 비교)
    walk_forward_evaluate(X, y, model, n_splits=5)

    print("프로세스 완료")
    print(df.columns)

if __name__ == "__main__":
    main()