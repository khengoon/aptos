import requests
import random
import os
import numpy as np
import cv2
from skimage import measure
import torch


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_eye = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_yrdv5ybi.json')


