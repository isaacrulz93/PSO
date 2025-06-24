import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from scipy.linalg import sqrtm, inv, logm, expm


class GLRSQ(BaseEstimator, ClassifierMixin):
    """
    GLRSQ는 리만기하학 기반의 공간분류기.
    레이블 수 만큼 M개의 프로토타입을 학습하고, 훈련데이터 Pi가 들어오면서 같은 클래스의 가장 가까운 프로토타입 WJ와 다른 클래스의 가장 가까운 프로토타입 WK를 찾는다.
    WJ와 WK의 거리를 dJ, dK라고 할 때, GLRSQ는 다음과 같은 로지스틱 스케일링을 사용하여 프로토타입 업데이트를 수행:
    E(Pi) = logistic((dK - dJ) / (dK + dJ))

    중요 포인트 : j와 k. 이는 같은 클래스와 다른 클래스의 프로토타입을 의미.
    GLRSQ의 목표는 각 클래스의 프로토타입을 업데이트하여, 같은 클래스의 프로토타입은 더 가까워지고, 다른 클래스의 프로토타입은 더 멀어지도록 하는 것.




    NumPy/SciPy-based GLRSQ with logistic scaling for prototype updates.
    Implements: E(Pi) = logistic((dK - dJ) / (dK + dJ))
    """
    def __init__(self, n_classes=4, n_prototypes=1, max_iter=5, learning_rate=0.1):
        self.n_classes = n_classes #클래스 개수
        self.n_prototypes = n_prototypes #클래스별 프로토타입 개수
        self.max_iter = max_iter # 최대 반복 횟수
        self.lr = learning_rate # 학습률

    def fit(self, X, y):
        self.prototypes_ = []
        self.labels_ = []
        classes = np.unique(y)

        for cls in classes:
            Xc = X[y == cls] # 클래스 cls에 해당하는 데이터만 선택
            proto = mean_riemann(Xc) # 평균으로 프로토타입 계산
            self.prototypes_.append(proto)
            self.labels_.append(cls)

        self.prototypes_ = np.stack(self.prototypes_)
        self.labels_ = np.array(self.labels_)

        for _ in range(self.max_iter):
            for Pi, yi in zip(X, y):
                dists = np.array([distance_riemann(Pi, W) for W in self.prototypes_]) # Pi와 각 프로토타입 W 사이의 거리 계산

                same_mask = (self.labels_ == yi)
                diff_mask = ~same_mask

                # if not np.any(same_mask) or not np.any(diff_mask):
                #     continue

                idx_J = np.argmin(dists[same_mask]) # 같은 클래스의 프로토타입 중 가장 가까운 것(J)
                idx_K = np.argmin(dists[diff_mask]) # 다른 클래스의 프로토타입 중 가장 가까운 것(K)

                J = np.where(same_mask)[0][idx_J] # 같은 클래스 프로토타입 인덱스
                K = np.where(diff_mask)[0][idx_K] # 다른 클래스 프로토타입 인덱스


                #선탠된 프로토타입
                WJ = self.prototypes_[J]
                WK = self.prototypes_[K]

                # Pi와 프로토타입 사이의 거리
                dJ = dists[J]
                dK = dists[K]

                #논문에 제시된 로지스틱 함수
                x = (dK - dJ) / (dK + dJ + 1e-8)
                omega = 1 / (1 + np.exp(-x))
                omega_p = omega * (1.0 - omega)  # ϖ'(x)

                # Cost 함수의 미분값 계산
                grad_J = - omega_p * (4.0 * dK) / ((dJ + dK + 1e-8) ** 2) #가까워지게 가고
                grad_K = + omega_p * (4.0 * dJ) / ((dJ + dK + 1e-8) ** 2) #멀어지게 가고

                #미분값은 각 프로토타입에(점 WJ, WK에 대한 업데이트 방향을 결정.)
                Vj = self._logmap(WJ, Pi)
                Vk = self._logmap(WK, Pi)

                #미분값 기반으로 프로토타입 업데이트
                self.prototypes_[J] = self._expmap(WJ, -self.lr * grad_J  * Vj)
                self.prototypes_[K] = self._expmap(WK, -self.lr * grad_K * Vk)

        return self

    def predict(self, X):
        preds = []
        for Pi in X:
            dists = np.array([distance_riemann(Pi, W) for W in self.prototypes_])
            pred = self.labels_[np.argmin(dists)]
            preds.append(pred)
        return np.array(preds)

    @staticmethod
    def _logmap(base, P):
        base_sqrt = sqrtm(base)
        base_inv_sqrt = inv(base_sqrt)
        return base_sqrt @ logm(base_inv_sqrt @ P @ base_inv_sqrt) @ base_sqrt

    @staticmethod
    def _expmap(base, V):
        base_sqrt = sqrtm(base)
        base_inv_sqrt = inv(base_sqrt)
        return base_sqrt @ expm(base_inv_sqrt @ V @ base_inv_sqrt) @ base_sqrt