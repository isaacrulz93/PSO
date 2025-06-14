import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from scipy.linalg import sqrtm, inv, logm, expm


class GLRSQ(BaseEstimator, ClassifierMixin):
    """
    NumPy/SciPy-based GLRSQ with logistic scaling for prototype updates.
    Implements: E(Pi) = logistic((dK - dJ) / (dK + dJ))
    """
    def __init__(self, n_classes=4, n_prototypes=1, max_iter=5, learning_rate=0.1):
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes
        self.max_iter = max_iter
        self.lr = learning_rate
        self._logmap_cache = {}

    def fit(self, X, y):
        self.prototypes_ = []
        self.labels_ = []
        classes = np.unique(y)

        for cls in classes:
            Xc = X[y == cls]
            proto = mean_riemann(Xc)
            self.prototypes_.append(proto)
            self.labels_.append(cls)

        self.prototypes_ = np.stack(self.prototypes_)
        self.labels_ = np.array(self.labels_)

        for _ in range(self.max_iter):
            for Pi, yi in zip(X, y):
                dists = np.array([distance_riemann(Pi, W) for W in self.prototypes_])

                same_mask = (self.labels_ == yi)
                diff_mask = ~same_mask

                # if not np.any(same_mask) or not np.any(diff_mask):
                #     continue

                idx_J = np.argmin(dists[same_mask])
                idx_K = np.argmin(dists[diff_mask])

                J = np.where(same_mask)[0][idx_J]
                K = np.where(diff_mask)[0][idx_K]

                WJ = self.prototypes_[J]
                WK = self.prototypes_[K]

                dJ = dists[J]
                dK = dists[K]

                # 수정 제안
                x = (dK - dJ) / (dK + dJ + 1e-8)
                omega = 1 / (1 + np.exp(-x))  # 시그모이드 함수 값 f(x)
                sigmoid_derivative = omega * (1 - omega)  # 시그모이드 함수의 미분 f'(x)

                # WJ와 WK에 대한 업데이트 스케일링 팩터가 원래는 다름
                # d(x)/dJ = 2*dK / (dJ+dK)^2
                # d(x)/dK = -2*dJ / (dJ+dK)^2
                # 하지만 단순화를 위해 하나의 grad_factor를 사용하는 경우가 많음. 현재 구현을 존중.
                # d(x)/dJ 와 d(x)/dK 를 곱해주는 부분이 grad_factor
                scalar_part_J = -(4 * dK) / ((dJ + dK + 1e-8) ** 2)
                scalar_part_K = (4 * dJ) / ((dJ + dK + 1e-8) ** 2)  # Note: 원본 논문에서는 K에 대한 미분은 -가 붙음. 부호는 밖에서 조절.

                # 업데이트
                Vj = self._logmap(WJ, Pi)
                Vk = self._logmap(WK, Pi)

                # WJ 업데이트
                update_J = -1 * self.lr * sigmoid_derivative * scalar_part_J * Vj
                self.prototypes_[J] = self._expmap(WJ, update_J)

                # WK 업데이트
                update_K = -1 * self.lr * sigmoid_derivative * scalar_part_K * Vk
                self.prototypes_[K] = self._expmap(WK, update_K)

        return self

    def predict(self, X):
        preds = []
        for Pi in X:
            dists = np.array([distance_riemann(Pi, W) for W in self.prototypes_])
            pred = self.labels_[np.argmin(dists)]
            preds.append(pred)
        return np.array(preds)


    def _logmap(self, base, P):
        key = base.tobytes()  # SPD matrix를 고유한 바이트 시그니처로 만듦
        if key not in self._logmap_cache:
            base_sqrt = sqrtm(base)
            base_inv_sqrt = inv(base_sqrt)
            self._logmap_cache[key] = (base_sqrt, base_inv_sqrt)
        else:
            base_sqrt, base_inv_sqrt = self._logmap_cache[key]

        return base_sqrt @ logm(base_inv_sqrt @ P @ base_inv_sqrt) @ base_sqrt

    @staticmethod
    def _expmap(base, V):
        base_sqrt = sqrtm(base)
        base_inv_sqrt = inv(base_sqrt)
        return base_sqrt @ expm(base_inv_sqrt @ V @ base_inv_sqrt) @ base_sqrt