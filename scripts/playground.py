import os
import cv2
import numpy as np
import pickle

from scipy.special import expit, logit

a = np.array([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 0.99, 0.999])

#print(a)

def f(p):
    r = p - 0.05
    r[r < 0] *= 20
    r[r > 0] += 0.05
    r *= 1.5
    r = expit(r)
    return r

def i(v):
    r = logit(v)
    r /= 1.5
    r[r > 0.05] -= 0.05
    r[r < 0] /= 20
    r += 0.05
    return r


b = f(a)
c = i(b)
np.set_printoptions(precision=3, suppress=True)
print(np.vstack((a, b, c)))


with open('selected_p_values.pkl', 'rb') as f:
    selected_p_values = pickle.load(f)

print(selected_p_values['Grow Multiple, 1'])

with open('matches.pkl', 'rb') as f:
    matches = pickle.load(f)
print(matches)