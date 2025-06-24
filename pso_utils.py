from pyriemann.classification import MDM
from GLRSQ import GLRSQ
from sklearn.metrics import cohen_kappa_score
import numpy as np

def threshold_mask(particles, threshold=0.6):
    """
    particles: (n_particles, n_dimensions) in [0,1]
    return: binary masks (n_particles, n_dimensions)
    """
    return (particles >= threshold).astype(int)

def apply_mask_to_spd(Ps, mask):
    """
    Ps: np.ndarray of shape (n_samples, d, d)
    mask: np.ndarray of shape (d,), binary mask
    returns: reduced SPD matrices of shape (n_samples, d', d')
    """
    idx = np.nonzero(mask)[0]
    return Ps[:, idx][:, :, idx]  # Ps[:, idx, :][:, :, idx]도 동일

def evaluate_fitness(X_train, y_train, X_val, y_val, original_dim, selected_dim, alpha=0.6):
    """
    Fitness = (1 - rho) * (selected_dim / original_dim) + rho * kappa
    """
    clf = GLRSQ(n_classes=len(np.unique(y_train)), max_iter=1, learning_rate=0.1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    kappa = cohen_kappa_score(y_val, y_pred)
    penalty = selected_dim / original_dim
    fitness = (1 - alpha) * penalty + alpha * kappa
    return fitness