import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import data_load, load_weather_data, load_cpi_data
from preprocessing import merge_data
from model import train_and_evaluate, plot_results, walk_forward_evaluate, evaluate_baselines, evaluate_final_holdout

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

    # (수정 이력) 이전에는 하이퍼파라미터 그리드서치와 Walk-Forward 검증을 df 전체로 돌리고,
    # 그 Walk-Forward 점수를 보면서 그리드 범위(max_depth 등)를 다시 좁히는 식으로 튜닝했음.
    # 그러면 최종 보고하는 Walk-Forward 점수가 "완전히 처음 보는 데이터"에 대한 점수가 아니라
    # 모델 선택 과정에 이미 한 번 참고된 점수가 되어버림(검증셋 과적합).
    # 마지막 3개월을 튜닝 과정 전체에서 격리한 최종 홀드아웃으로 떼어놓고,
    # 그리드서치/Walk-Forward는 전부 그 이전 구간(dev set)에서만 수행하도록 분리함.
    cutoff_date = df['날짜'].max() - pd.DateOffset(months=3)
    dev_df = df[df['날짜'] < cutoff_date].reset_index(drop=True)
    final_holdout_df = df[df['날짜'] >= cutoff_date].reset_index(drop=True)
    print(f"   튜닝용 dev set: {len(dev_df)}일 ({dev_df['날짜'].min().date()} ~ {dev_df['날짜'].max().date()})")
    print(f"   최종 홀드아웃: {len(final_holdout_df)}일 ({final_holdout_df['날짜'].min().date()} ~ {final_holdout_df['날짜'].max().date()}, 튜닝 과정 어디에도 쓰이지 않음)")

    X_dev, y_dev = dev_df[features], dev_df['일매출']
    X_final, y_final = final_holdout_df[features], final_holdout_df['일매출']

    # dev set 안에서만 시간 순서를 보존한 채 마지막 20%를 테스트셋으로 고정
    # (shuffle=True를 쓰면 테스트 시점의 인접 날짜가 학습셋에 섞여 들어가 미래 정보 누수가 발생함)
    X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.2, shuffle=False)

    print("3. 모델 학습 및 하이퍼파라미터 탐색 중 (dev set 내부)...")
    model, pred = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("4. 베이스라인 대비 성능 비교 중 (dev set 홀드아웃 기준)...")
    # XGBoost 모델이 "아무 규칙 없이 어제 매출/7일 평균을 그대로 예측한 것"보다
    # 실제로 더 나은지 확인 (R2/MAE 단독 수치만으로는 모델의 가치를 판단할 수 없기 때문)
    evaluate_baselines(X_test, y_test)

    print("5. 예측 결과 시각화 중...")
    plot_results(model, y_test, pred, features)

    print("6. Walk-Forward 다중 폴드 검증 진행 중 (dev set 내부)...")
    # 마지막 20% 단일 홀드아웃이 특정 시즌에 쏠리는 문제를 보완하기 위해
    # 동일 하이퍼파라미터로 여러 시점을 기준으로 재검증 (베이스라인도 폴드마다 함께 비교).
    # 주의: 이 점수는 하이퍼파라미터를 고르면서 참고했던 지표라 낙관 편향이 섞여 있음 (README 참고).
    walk_forward_evaluate(X_dev, y_dev, model, n_splits=5)

    print("7. 최종 홀드아웃 검증 (튜닝에 전혀 관여하지 않은 마지막 3개월, 단 한 번만 평가)...")
    evaluate_final_holdout(model, X_dev, y_dev, X_final, y_final)

    print("프로세스 완료")
    print(df.columns)

if __name__ == "__main__":
    main()