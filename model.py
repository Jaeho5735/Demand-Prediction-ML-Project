from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
    param_grid = {
        'n_estimators': [500, 700, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }

    # GridSearchCV 설정 (TimeSeriesSplit 기반 3-Fold 교차 검증)
    # 시계열 데이터이므로 각 폴드의 학습 구간이 항상 검증 구간보다 과거여야 함.
    # 기본 KFold(shuffle=False)는 구간을 시간순으로 자르긴 하지만, 앞쪽 구간을 검증하는
    # 폴드에서는 뒤쪽(미래) 구간이 학습에 쓰여 미래 정보가 유입되는 문제가 있어 TimeSeriesSplit로 대체.
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='r2',
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

def walk_forward_evaluate(X, y, base_model, n_splits=5):
    """
    마지막 20% 단일 홀드아웃 평가는 그 구간이 우연히 특이 시즌(예: 연말)에
    쏠릴 경우 결과가 왜곡될 수 있다. TimeSeriesSplit으로 여러 시점을 기준으로
    학습/검증을 반복(walk-forward)하여 더 안정적인 성능 추정치를 얻는다.
    base_model과 동일한 하이퍼파라미터로 각 폴드마다 새로 학습한다.
    """
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
        fold_results.append({'fold': fold_idx, 'train_size': len(X_train), 'test_size': len(X_test), 'r2': r2, 'mae': mae})
        print(f"Fold {fold_idx}: train={len(X_train)}, test={len(X_test)} | R2={r2:.4f} | MAE={mae:,.0f}원")

    r2_values = [f['r2'] for f in fold_results]
    mae_values = [f['mae'] for f in fold_results]
    print(f"\n평균 R2: {np.mean(r2_values):.4f} (표준편차 {np.std(r2_values):.4f})")
    print(f"평균 MAE: {np.mean(mae_values):,.0f}원 (표준편차 {np.std(mae_values):,.0f}원)")

    return fold_results

def plot_results(model, y_test, y_pred, feature_names):
    """
    예측 결과 및 변수 중요도를 시각화합니다.
    """
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