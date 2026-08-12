from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.base import clone
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import platform

# 모델 학습 및 평가 함수
def train_and_evaluate(X_train, X_test, y_train, y_test):
    xgb = XGBRegressor(
        n_jobs=-1,
        random_state=42
    )

    # 탐색할 파라미터 그리드 설정
    # (수정 이력) 기존 그리드(n_estimators 500~1000, max_depth 3~7, 정규화 파라미터 없음)는
    # 학습 표본 규모(홀드아웃 기준 269행, Walk-Forward 1폴드는 57행)에 비해 지나치게 복잡한 쪽으로만
    # 탐색해서 과적합을 유도했음 (실측: 당시 best_params였던 depth=7·n=500 조합은 Train R2=0.979,
    # Test R2=0.040으로 과적합 갭이 매우 컸고, Walk-Forward 5-Fold에서 이동평균 베이스라인에 5전 5패).
    # depth를 낮출수록 Walk-Forward 성능이 꾸준히 개선되는 것을 확인해, 얕은 트리(max_depth 1~3)와
    # 적은 트리 수(n_estimators 30~200), 정규화 파라미터(reg_lambda, min_child_weight)를 탐색 범위에
    # 새로 포함시킴. 조합 수가 커져(4*3*3*3*3*3=972) 전수 탐색인 GridSearchCV 대신
    # RandomizedSearchCV(n_iter=150)로 전환함.
    param_distributions = {
        'n_estimators': [30, 50, 100, 200],
        'learning_rate': [0.03, 0.05, 0.1],
        'max_depth': [1, 2, 3],
        'subsample': [0.7, 0.8, 0.9],
        'reg_lambda': [1, 5, 10],
        'min_child_weight': [1, 5, 10],
    }

    # RandomizedSearchCV 설정 (TimeSeriesSplit 기반 3-Fold 교차 검증)
    # 시계열 데이터이므로 각 폴드의 학습 구간이 항상 검증 구간보다 과거여야 함.
    # 기본 KFold(shuffle=False)는 구간을 시간순으로 자르긴 하지만, 앞쪽 구간을 검증하는
    # 폴드에서는 뒤쪽(미래) 구간이 학습에 쓰여 미래 정보가 유입되는 문제가 있어 TimeSeriesSplit로 대체.
    grid_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        n_iter=150,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='r2',
        random_state=42,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    print(f"최적 파라미터: {grid_search.best_params_}")

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print(f"--- 모델 평가 결과 ---")
    print(f"R2 Score (설명력): {r2:.4f} (1에 가까울수록 완벽)")
    print(f"MAE (평균 절대 오차): {mae:,.0f}원")
    print(f"RMSE (제곱근 평균 제곱 오차): {rmse:,.0f}원")

    return model, pred


def _score_baseline(y_true, y_pred_baseline, label):
    r2 = r2_score(y_true, y_pred_baseline)
    mae = mean_absolute_error(y_true, y_pred_baseline)
    print(f"  [베이스라인] {label:<28} R2={r2:>7.4f} | MAE={mae:>12,.0f}원")
    return {'label': label, 'r2': r2, 'mae': mae}


def evaluate_baselines(X_test, y_test, label='홀드아웃 테스트셋 기준'):
    print(f"\n--- 베이스라인 비교 ({label}) ---")
    results = []
    results.append(_score_baseline(y_test, X_test['전일매출'], 'Naive (전일매출)'))
    results.append(_score_baseline(y_test, X_test['7일평균매출'], '이동평균 (7일평균매출)'))
    return results


def evaluate_final_holdout(base_model, X_dev, y_dev, X_final, y_final):
    # 하이퍼파라미터 탐색, Walk-Forward 검증 어디에도 이 구간(마지막 3개월)의 점수를
    # 참고한 적이 없으므로, 여기서 나오는 값이 이 프로젝트에서 낙관 편향이 가장 적은 지표다.
    # 같은 하이퍼파라미터로 dev set 전체(튜닝에 쓰인 80%+20% 전부)를 다시 학습시킨 뒤,
    # 최종 홀드아웃에서 딱 한 번만 예측/평가한다.
    final_model = clone(base_model)
    final_model.fit(X_dev, y_dev)
    pred = final_model.predict(X_final)

    r2 = r2_score(y_final, pred)
    mae = mean_absolute_error(y_final, pred)
    rmse = np.sqrt(mean_squared_error(y_final, pred))

    print("\n--- 최종 홀드아웃 검증 (튜닝에 전혀 관여하지 않은 마지막 3개월) ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:,.0f}원")
    print(f"RMSE: {rmse:,.0f}원")

    evaluate_baselines(X_final, y_final, label='최종 홀드아웃 기준')

    return final_model, pred, {'r2': r2, 'mae': mae, 'rmse': rmse}


def walk_forward_evaluate(X, y, base_model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    print(f"\n--- Walk-Forward 검증 ({n_splits}-Fold, TimeSeriesSplit) ---")
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = clone(base_model)
        fold_model.fit(X_train, y_train)
        pred = fold_model.predict(X_test)

        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)

        # 같은 폴드에서 베이스라인도 함께 계산 (모델과 동일한 기준으로 비교하기 위함)
        naive_r2 = r2_score(y_test, X_test['전일매출'])
        naive_mae = mean_absolute_error(y_test, X_test['전일매출'])
        ma_r2 = r2_score(y_test, X_test['7일평균매출'])
        ma_mae = mean_absolute_error(y_test, X_test['7일평균매출'])

        fold_results.append({
            'fold': fold_idx, 'train_size': len(X_train), 'test_size': len(X_test),
            'r2': r2, 'mae': mae,
            'naive_r2': naive_r2, 'naive_mae': naive_mae,
            'ma_r2': ma_r2, 'ma_mae': ma_mae,
        })
        print(f"Fold {fold_idx}: train={len(X_train)}, test={len(X_test)}")
        print(f"  [XGBoost]          R2={r2:>7.4f} | MAE={mae:>12,.0f}원")
        print(f"  [Naive 베이스라인]   R2={naive_r2:>7.4f} | MAE={naive_mae:>12,.0f}원")
        print(f"  [이동평균 베이스라인] R2={ma_r2:>7.4f} | MAE={ma_mae:>12,.0f}원")

    r2_values = [f['r2'] for f in fold_results]
    mae_values = [f['mae'] for f in fold_results]
    naive_r2_values = [f['naive_r2'] for f in fold_results]
    ma_r2_values = [f['ma_r2'] for f in fold_results]

    print(f"\n평균 R2 (XGBoost): {np.mean(r2_values):.4f} (표준편차 {np.std(r2_values):.4f})")
    print(f"평균 MAE (XGBoost): {np.mean(mae_values):,.0f}원 (표준편차 {np.std(mae_values):,.0f}원)")
    print(f"평균 R2 (Naive 베이스라인): {np.mean(naive_r2_values):.4f}")
    print(f"평균 R2 (이동평균 베이스라인): {np.mean(ma_r2_values):.4f}")

    return fold_results


def plot_results(model, y_test, y_pred, feature_names):
    # 예측 결과 및 변수 중요도 시각화
    # 한글 폰트 설정 (OS별 대응)
    if platform.system() == 'Darwin': # macOS
        plt.rc('font', family='AppleGothic')
    else: # Windows/Linux
        plt.rc('font', family='Malgun Gothic')
    plt.rc('axes', unicode_minus=False)

    # 1. 변수 중요도 시각화 (Top 10)
    plt.figure(figsize=(10, 8))
    feat_importances = pd.Series(model.feature_importances_, index=feature_names)
    feat_importances.nlargest(10).sort_values().plot(kind='barh', color='skyblue')
    plt.title('매출 예측에 기여한 주요 변수 (Feature Importance)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

    # 2. 실제 매출 vs 예측 매출 비교 그래프
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.values, label='Actual Sales (실제)', marker='o', alpha=0.7)
    plt.plot(y_pred, label='Predicted Sales (예측)', marker='x', alpha=0.7)
    plt.title('실제 매출 vs XGBoost 예측 매출 비교')
    plt.xlabel('Time (Test Data Index)')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()