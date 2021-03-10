import numpy as np
import pandas as pd
# %matplotlib inline
from matplotlib import pyplot as plt
import os
import sys

def closest(target, lst):
  diff = sys.maxsize
  sol = 0;
  for i in lst:
    temp = abs(i - target)
    if (temp < diff):
      diff = temp;
      sol = i;
  return sol;

def kij_func(t, s, spxw_call_clean):
  kij = []
  for j in t:
    # k_j represent the strikes at jth maturity
    kj = []
    # finding the closest maturity/strike if no particular data
    t_temp = spxw_call_clean['okey_maturity'].unique()
    j_temp = closest(j, t_temp);
    iv = spxw_call_clean[(spxw_call_clean['okey_maturity'] == j_temp)]['prtIv'].mean()
    n = len(s)
    # k_j^i represent the ith strike at jth maturity
    for i in range(n):
      kj.append(np.round(np.exp((3 * ((2 *(i + 1) - (n - 1)) / (n + 1)) * iv * np.sqrt(j)) - 0.5 * iv *iv * j), 2))
    kij.append(kj)
  return np.asarray(kij)

def thetaij_func(t, s, spxw_call_clean):
  thetaij = []
  # theta_j represent the theta at jth maturity
  for j in t:
    thetaj = []
    # finding the closest maturity if no particular data
    t_temp = spxw_call_clean['okey_maturity'].unique()
    j_temp = closest(j, t_temp);
     # theta_j^i represent the ith theta at jth maturity
    for i in s:
      # finding the closest strike if no particular data
      s_temp = np.asarray(spxw_call_clean[(spxw_call_clean['okey_maturity'] == j_temp)]['okey_xx'])
      i_temp = closest(i, s_temp)
      
      thetaj.append(-1 * np.asarray(spxw_call_clean[(spxw_call_clean['okey_maturity'] == j_temp) & (spxw_call_clean['okey_xx'] == i_temp)]['prtTh'])[0])
    thetaij.append(thetaj)
  return np.asarray(thetaij)

def gammaij_func(t, s, spxw_call_clean):
  gammaij = []
  # gamma_j represent the gamma at jth maturity
  for j in t:
    gammaj = []
    # finding the closest maturity if no particular data
    t_temp = spxw_call_clean['okey_maturity'].unique()
    j_temp = closest(j, t_temp)
    # gamma_j^i represent the ith theta at jth maturity
    for i in s:
      # finding the closest strike if no particular data
      s_temp = np.asarray(spxw_call_clean[(spxw_call_clean['okey_maturity'] == j_temp)]['okey_xx'])
      i_temp = closest(i, s_temp)

      gammaj.append(np.asarray(spxw_call_clean[(spxw_call_clean['okey_maturity'] == j_temp) & (spxw_call_clean['okey_xx'] == i_temp)]['prtGa'])[0])

    gammaij.append(gammaj)
  return np.asarray(gammaij)

def tj_func(t):
  tj = []
  # calculating the time difference
  for i in range(len(t) - 1):
    tj.append(t[i + 1] - t[i])
  return np.asarray(tj)

def dlv_func(thetaij, gammaij, kij, tj, t, s, spxw_call_clean):
  dlvij = []
  for j in range(len(tj)):
    dtj = tj[j]
    dlvj = []
    for i in range(len(s)):
      if gammaij[j][i] != 0 and kij[j][i] != 0:
        dlvj.append(
              np.sqrt(2 * thetaij[j][i] / (gammaij[j][i] * kij[j][i] * kij[j][i] * dtj))
        )
      else:
        dlvj.append(0)
    dlvij.append(dlvj)
  return np.asarray(dlvij)

directory_txt = os.fsencode('/data/txt/')
DF_list = list()
spxw_call_data = 0;
    
for file in os.listdir(directory_txt):
  filename = os.fsdecode(file)
  if filename.endswith(".txt"):
    date = filename[-14:-4]
    print(date)
    data = pd.read_csv('/data/txt/' + filename, sep="\t")
    EQT = pd.DataFrame(data)
    # Import and Create Fields
    df = EQT[['okey_tk', 
              'okey_yr', 'okey_mn', 'okey_dy', 
              'okey_xx', 
              'okey_cp', 
              'prtSize', 'prtPrice',
              'prtIv', 
              'prtGa', 'prtTh', 'surfOpx']]
    df['okey_ymd'] = pd.to_datetime(df['okey_yr'].astype(str) + '/' + df['okey_mn'].astype(str) + '/' + df['okey_dy'].astype(str))
    df['okey_maturity'] = df['okey_ymd'] - np.datetime64(date)
    df['okey_maturity'] = df['okey_maturity'].dt.days
    df = df.drop_duplicates().sort_values(by=['okey_maturity'])
    
    # SPX Weekly Call 
    spxw_call = df.loc[(df['okey_tk'] == 'SPXW') & (df['okey_cp'] == 'Call')]
    spxw_call = spxw_call[['okey_xx',  
              'okey_maturity',
              'prtSize', 'prtPrice',
              'prtIv', 
              'prtGa', 'prtTh', 'surfOpx']]
    grouped_df = spxw_call.groupby(['okey_maturity', 'okey_xx'])
    spxw_call_clean = grouped_df.mean().reset_index() 
    spxw_call_data += spxw_call_clean.shape[0]
    plt.scatter(spxw_call_clean['okey_xx'], spxw_call_clean['okey_maturity'])
    plt.xlabel('option price')
    plt.ylabel('date')
    DF_list.append(spxw_call_clean)

    # time = np.sort(spxw_call_clean['okey_maturity'].unique()/365)
    # strike = spxw_call_clean.groupby(['okey_xx']).size().reset_index(name='count')
    # s = np.sort(np.asarray(strike.loc[strike['count'] > 20]['okey_xx']))
    # print('maturity', time * 365)
    # print('strike', s)
plt.show()

t = [20, 50, 71, 106]
s = [3230, 3240, 3250, 3260, 3270, 3280, 3290, 3300]

dlv = []
for i in range(0, len(DF_list)):
  spxw_call_clean = DF_list[i]
  kij = kij_func(t, s, spxw_call_clean)
  thetaij = thetaij_func(t, s, spxw_call_clean)
  gammaij = gammaij_func(t, s, spxw_call_clean)
  tj = tj_func(t)
  dlv_ind = dlv_func(thetaij, gammaij, kij, tj, t, s, spxw_call_clean)
  print(dlv_ind)
  dlv.append(dlv_ind)
