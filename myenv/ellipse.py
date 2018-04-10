import numpy as np
import numpy.linalg as linalg
import math
from numpy import pi

def ellipse(mu, cov, n=100, conf_int=5.991):
    w, v = linalg.eigh(cov)
    max_eig = v[:, -1]
    phi = math.atan2(max_eig[1], max_eig[0])
    phi %= 2 * pi
    chi2_val = np.sqrt(conf_int) # 95% confidence interval
    a =  chi2_val * np.sqrt(w[1])
    b =  chi2_val * np.sqrt(w[0])
    #theta = np.arange(0, 2 * pi, 0.01)
    theta = np.linspace(0, 2 * pi, n)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    R = np.array([[np.cos(phi), np.sin(phi)],
                  [-np.sin(phi), np.cos(phi)]])
    X = np.array([x,y]).T
    X = X.dot(R)
    x = X[:,0] + mu[0]
    y = X[:,1] + mu[1]
    return x, y
