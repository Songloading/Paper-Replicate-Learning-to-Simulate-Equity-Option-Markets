import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import skew,  kurtosis

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
s = find('*.csv','data/csv/')
r = find('*.csv','result/')

def compare_epdf(a, b):
    ecdfa = ECDF(a)
    ecdfb = ECDF(b)
    return ecdfa, ecdfb

def euclidean_distance(a, b):
    # a and b should be in same size
    sum = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            sum = sum +  np.square(a[i,j] - b[i, j])
    return np.sqrt(sum)
# load data
filelength = (len(s))
dt_arry = np.zeros((filelength, 3, 8))
i = 0
for f in s:
    dt_arry[i,:,:] = np.loadtxt(open(f, "rb"), delimiter=",")
    i = i+1
dt_arry = dt_arry.reshape(filelength*3*8)

rs_arry = np.zeros((filelength, 3, 8))
i = 0
for f in r:
    rs_arry[i,:,:] = np.loadtxt(open(f, "rb"), delimiter=",")
    i = i+1
rs_arry = rs_arry.reshape(filelength*3*8)
plt.hist(rs_arry,  alpha=0.5, label='simulated')
plt.hist(dt_arry,  alpha=0.5, label='real')
plt.legend(loc='upper right')
plt.show()

# calculate difference in cumulative epdf
i = 0
c = 0
lim = int(filelength/5-1)
for i in range(0,lim):
    suba = rs_arry[i*24:(i+1)*24]
    subb = dt_arry[i*24:(i+1)*24]
    max = np.max(subb)
    ecdfa, ecdfb = compare_epdf(suba, subb)
    c = c + abs(ecdfa(max) - ecdfb(max))
print('cumulative ecdf: ' + str(c))

skewR = skew(dt_arry)
skewF = skew(rs_arry)
print('skew diff: ' + str(abs(skewR-skewF)/90))

kurR = kurtosis(dt_arry)
kurF = kurtosis(rs_arry)
print('kurtosis diff: ' + str(abs(kurR-kurF)/filelength))

# rs_arry = rs_arry.reshape((filelength,3,8))
# dt_arry = dt_arry.reshape((filelength, 3, 8))

# min_eu = 100
# for i in range(filelength):
#     no = euclidean_distance(rs_arry[i,:,:], dt_arry[i,:,:])
#     if no< min_eu:
#         min_eu = no
# print('min corrrelation: ' + str(min_eu))
