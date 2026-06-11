from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
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

    # GridSearchCV 설정 (3-Fold 교차 검증)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
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