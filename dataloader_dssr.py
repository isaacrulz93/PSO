import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
import pandas as pd
import os

# 경로 설정 (절대경로)


def load_bnci2014_001():
    """Load and preprocess BNCI2014_001 from MOABB"""
    paradigm = MotorImagery(n_classes=4, resample=128)
    dataset = BNCI2014_001()
    X, y, meta = paradigm.get_data(dataset)
    return X, y, meta

def load_dataset(name, subject_id=None):
    """
    General dataset loader for DSSR framework.
    name: str, one of ['BNCI2014_001', 'IIIa']
    Returns: X, y, meta
    """
    if name == 'BNCI2014_001':
        return load_bnci2014_001()