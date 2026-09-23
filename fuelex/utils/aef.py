import numpy as np

def dequantize(aef_arr):
    return ((aef_arr / 127.5) ** 2) * np.sign(aef_arr)