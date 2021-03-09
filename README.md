# Deep Hedging: Learning to Simulate Equity Option Markets

Welcome to our paper replicate project- [Deep Hedging: Learning to Simulate Equity Option Markets](https://arxiv.org/pdf/1911.01700.pdf) by Song Luo and Yunshu Zhang. 

This paper is divided into seven parts; they are:

- Description fo the Problem
- Shortcoming of the Eisting Methods
- Exploratory Data Analysis
- Gans Model
- Results and Evaluation
- Drawback and Challenge
- Future Study

### **Description of the Problem**

There is growing interest in applying reinforcement learning techniques to the problem of managing a portfolio of derivatives. This involves not only the underlying assets but also the available exchange-traded options. Hence, in order to train an options trading model, we need more time-series data that includes option prices. The reason why the simulator is a hot topic today. 

- The amount of useful real-life data available is limited. This motivates the need for a realistic simulator. 
- This paper constructs realistic equity option market simulators based on generative adversarial networks. 
- This is the first time that GANs can be applied to the task of generating multivariate financial time series.

### **Shortcoming of the Existing Methods**

Lack of option data is already a long-term issue. But none of the models have been used in this area. 

- **Classical derivative pricing model** also require generators like GANs, but these are **not realistic**. They typically limited to a small number of driving factors.
- **PCA** focus on only **implied volatility data**.
- **Risk-neutral market model** focus on ensuring the martingale property **rather than realistic**.
- **GANs** have been used to create/generate realistic synthetic time series for **asset prices**. 

As we notices, none of them focusing on the realistic. Although GANs could generate realistic synthetic, only apply on asset prices. Neural networks has not been applied to option market generation yet.

### **Exploratory Data Analysis**

We are using the option data from SpiderRock. And we are generating option data of SPX weekly calls from January, February, March, June, and July 2020.

##### Data

According to the dummy data, it contains 88 fields , including every option print along with quote, surface, and so on at print time. Below is the head of the dataset.

| okey_ts | okey_tk | okey_yr | okey_mn | okey_dy | okey_xx | okey_cp | prtNumber  | ticker_ts | ticker_tk | prtExch | prtSize | prtPrice | prtType | prtOrders | prtClusterNum | prtClusterSize | prtVolume | cxlVolume | bidCount | askCount | bidVolume | askVolume | ebid | eask | ebsz | easz | eage | prtSide | prtTimestamp | netTimestamp | timestamp | oBid | oAsk | oBidSz | oAskSz | oBidEx | oAskEx | oBidExSz | oAskExSz | oBidCnt | oAskCnt | uPrc       | yrs    | rate   | sdiv   | ddiv | oBid2 | oAsk2 | oBidSz2 | oAskSz2 | uBid       | uAsk       | xDe    | xAxis  | prtIv  | prtDe | prtGa  | prtTh   | prtVe  | prtRo  | calcErr | surfVol | surfOpx | surfAtm | prtProbability | oBidM1 | oAskM1 | uBidM1     | uAskM1     | uPrcM1     | sVolM1 | sOpxM1 | sDivM1 | sErrM1 | pnlM1  | pnlM1Err | oBidM10 | oAskM10 | uBidM10    | uAskM10    | uPrcM10    | sVolM10 | sOpxM10 | sDivM10 | sErrM10 | pnlM10 | pnlM10Err |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ---------- | --------- | --------- | ------- | ------- | -------- | ------- | --------- | ------------- | -------------- | --------- | --------- | -------- | -------- | --------- | --------- | ---- | ---- | ---- | ---- | ---- | ------- | ------------ | ------------ | --------- | ---- | ---- | ------ | ------ | ------ | ------ | -------- | -------- | ------- | ------- | ---------- | ------ | ------ | ------ | ---- | ----- | ----- | ------- | ------- | ---------- | ---------- | ------ | ------ | ------ | ----- | ------ | ------- | ------ | ------ | ------- | ------- | ------- | ------- | -------------- | ------ | ------ | ---------- | ---------- | ---------- | ------ | ------ | ------ | ------ | ------ | -------- | ------- | ------- | ---------- | ---------- | ---------- | ------- | ------- | ------- | ------- | ------ | --------- |
| NMS     | SPXW    | 2020    | 2       | 10      | 3355    | Call    | 1.1014E+18 | CBOE      | SPX       | CBOE    | 25      | 6.2      | 73      | 0         | 1             | 25             | 25        | 0         | 0        | 0        | 0         | 0         | 6    | 6.4  | 39   | 4    | -1   | Mid     | 1.5811E+18   | 1.5811E+18   | 00:03.4   | 6    | 6.4  | 39     | 4      | CBOE   | CBOE   | 39       | 4        | 1       | 1       | 3344.61304 | 0.0086 | 0.0182 | 0.0477 | 0    | 0     | 0     | 0       | 0       | 3344.48804 | 3344.73804 | 0.1534 | 0.3308 | 0.0882 | 0.342 | 0.0134 | -2.4852 | 1.1365 | 0.0975 | None    | 0.0893  | 6.3188  | 0.0957  | 0              | 5.7    | 6.1    | 3343.23804 | 3343.48804 | 3343.24304 | 0.0894 | 5.8712 | 0.0483 | None   | 0.1398 | No       | 5.8     | 6.2     | 3343.48804 | 3343.73804 | 3343.73304 | 0.0893  | 6.0421  | 0.0458  | None    | 0.143  | No        |

We find that these 10 fields are extremely useful for us:

- okey_tk: Option symbol
- okey_yr: Option expiration year
- okey_mn: Option expiration month
- okey_dy: Option expiration day
- okey_xx: Option strike
- okey_cp: Option call
- prtlv: Print implied vol
- prtGa: Print gamma
- prtTh: Print theta
- surfOpx: SR surface price

```python
EQT = pd.DataFrame(data)
df = EQT[['okey_tk', 
	'okey_yr', 'okey_mn', 'okey_dy', 
	'okey_xx', 
	'okey_cp', 
	'prtSize', 'prtPrice',
	'prtIv', 
	'prtGa', 'prtTh', 'surfOpx']]
```

Then based on the expiration date, we calculating the maturity.

```python
df['okey_ymd'] = pd.to_datetime(df['okey_yr'].astype(str) + '/'
                  + df['okey_mn'].astype(str)
                  + '/' + df['okey_dy'].astype(str))
df['okey_maturity'] = df['okey_ymd'] - np.datetime64(date)
df['okey_maturity'] = df['okey_maturity'].dt.days
```

##### Choosing Strike and Maturity

Since not all the strikes and maturity are evenly distributed, and we want to avoid all the 0 entry for our matrix.

![img](https://lh4.googleusercontent.com/a7T_IXfjw4pNunlvus34cB8lolUQZHcwUsjdLVXpaiDdGsbPDnCx-sM5a10SMqThK7qm1e-LbYHYMrDef9Uxr8UeVpRdsA38gWi34hZN7v-QgvmYTPaeQxxUZvKwkd2dlplrFLHX-iM)

Aftering plotting strike vs maturity, we decided use this set of K and M.

**K = [3230, 3240, 3250, 3260, 3270, 3280, 3290, 3300]**

**M = [20, 50, 71, 106]**

We denote Nk as strike K's dimension; Nm as maturity M's dimentsion.

##### Discrete Local Volatility (DLV)

###### What is DLV

Discrete Local Volatility (DLV) also stand for discrete “Backward Local Volatility” which is a discrete version of [Dupire’s Local Volatility](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.320.5063&rep=rep1&type=pdf). . DLV converges to the proper Dupire models as the strike and maturity grid increases in density, and therefore retains its intuitive features.

###### Why DLV

Option prices are subject to strict ordering constraints because of no-arbitrage consideration. For example, since the call option payoff is a non-increasing function of strike, the option price must also be non-increasing. The violation of this rule would constitute an arbitrage opportunity. Because of this, it is necessary to convert plain data into DLV $N_k * N_m$ dimension.

###### How to Compute DLV

Regarding how to compute DLV, here is the formular.

![img](https://lh4.googleusercontent.com/LzdZwHFXIdidXV8XM8bYY6aXMKCm5fWADE3REX0a_yIXQBtRk20AzWMRtSfWkRz5r1NzLWFm3PYuPRn-JL8hoMcXvSaT8t_EZyf2-QQBjWRkI_BJcLhnHODV0-KhBSLw1Rtll6ZBtDo)

Since not all of the date has contains the specific strike set and maturity set, we decide to choose the closest strike price or maturity.

```python
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
```

**Maturities** (![img](https://latex.codecogs.com/png.latex?%5Cinline%20d_%7Bt%7D)): 

We define 

![img](https://latex.codecogs.com/png.latex?%5Cinline%20d_%7Bt&plus;j%7D%5E&plus;%20%3A%3D%20t_%7Bj&plus;1%7D-t_%7Bj%7D)

and

![img](https://latex.codecogs.com/png.latex?%5Cinline%20d_%7Bt&plus;j%7D%5E-%20%3A%3D%20t_%7Bj%7D-t_%7Bj-1%7D)

```python
def tj_func(t):
  tj = []
	# calculating the time difference
  for i in range(len(t) - 1):
    tj.append(t[i + 1] - t[i])
  return np.asarray(tj)
```

**Strikes** (![img](https://latex.codecogs.com/png.latex?%5Cinline%20k_%7Bj%7D%5E%7Bi%7D)):

We refer to the nj strikes of the jth maturity as (![img](https://latex.codecogs.com/png.latex?%5Cinline%20k_%7Bj%7D%3D%28k_%7Bj%7D%5E%7B-1%7D%2Ck_%7B0%7D%5E%7Bj%7D%2C...%2Ck_%7Bj%7D%5E%7Bn_%7Bj%7D%7D%29).

Because of requiring of increasing strikes, an practical remedy is “Jordinson scaling”, which to scale the process by its growth rate,or, equivalently, by providing a widening grid for the process. We denote this as ![img](https://latex.codecogs.com/png.latex?%5Cinline%20k_%7Bj%7D%5E%7Bi%7D).

```python
def kij_func(t, s, spxw_call_clean):
  kij = []
	# k_j represent the strikes at jth maturity
  for j in t:
    # kj represent the strikes at jth time
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
```

Both ![img](https://latex.codecogs.com/png.latex?%5Cinline%20%5CGamma_%7Bj%7D%5E%7Bi%7D) and ![img](https://latex.codecogs.com/png.latex?%5Cinline%20b%5CTheta%20_%7Bj%7D%5E%7Bi%7D) are define in the dataset. Although no need to calculate them, we have to put them together.

```python
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
```

```python
def gammaij_func(t, s, spxw_call_clean):
  gammaij = []
  # gamma_j represent the gamma at jth maturity
  for j in t:
    gammaj = []
    # finding the closest maturity if no particular data
    t_temp = spxw_call_clean['okey_maturity'].unique()
    j_temp = closest(j, t_temp);
    # gamma_j^i represent the ith theta at jth maturity
    for i in s:
      # finding the closest strike if no particular data
      s_temp = np.asarray(spxw_call_clean[(spxw_call_clean['okey_maturity'] == j_temp)]['okey_xx'])
      i_temp = closest(i, s_temp)
      
      gammaj.append(np.asarray(spxw_call_clean[(spxw_call_clean['okey_maturity'] == j_temp) & (spxw_call_clean['okey_xx'] == i_temp)]['prtGa'])[0])
    gammaij.append(gammaj)

  return np.asarray(gammaij)
```

Last but not the least, puting all together.

```python
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
```

And here is a head of our example DLV.

![img](/Users/joycefeifei/Library/Application Support/typora-user-images/image-20210309011145177.png)

In order to dive more into this how does the DLV come from, please read through [Discrete Local Volatility for Large Time Steps (Short Version)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2783409), [Arbitrage-free market models for option prices](http://www.nccr-finrisk.uzh.ch/media/pdf/wp/WP428_D1.pdf).

### **GANs Model**

##### Problem Formulation

###### Compute DLV

As we illustrate above, compute DLV based on K and M, and denote it as ![img](https://latex.codecogs.com/png.latex?%5Cinline%20%5Csigma_%7Bt%7D).

![img](https://latex.codecogs.com/png.latex?%5Cinline%20%5Csigma%3D%20%5B%5Csigma_%7Bt%7D%28K%2CM%29%5D_%7B%28K%2CM%29%5Cin%20K%20%5Ctimes%20M%7D%29%5D%2C%20t%20%5Cin%20%5Cmathbb%7BN%7D_%7B0%7D)



###### Regular Time Series

Base on the previous sigma function. We could easily generate the plain time-series formula. This formula involved mapping function g, noise Z, and state S. We assume that the historical process will evolves through a conditional model in the time series problem. The mapping function g relates noise and state to the next time step, in such form

![img](https://latex.codecogs.com/png.latex?%5Cinline%20%5Csigma_%7Bt&plus;1%7D%3Dg%28Z_%7Bt&plus;1%7D%2CS_t%29%2Ct%20%5Cin%20%5Cmathbb%7BN%7D_%7B0%7D)

where

![img](https://latex.codecogs.com/png.latex?%5Cinline%20g%3AL%5E%7B2%7D%28%5Cmathbb%7BR%7D%5E%7BN_%7BZ%7D%7D%29%5Ctimes%20L%5E%7B2%7D%28%5Cmathbb%7BR%7D%5E%7BN_%7BS%7D%7D%29%5Crightarrow%20L%5E%7B2%7D%28%5Cmathbb%7BR%7D%5E%7BN_%7BK%7D%5Ctimes%20N_%7BM%7D%7D%29)

![img](https://latex.codecogs.com/png.latex?%5Cinline%20Z_%7Bt&plus;1%7D%5Csim%20N%280%2CI%29)

![img](https://latex.codecogs.com/png.latex?%5Cinline%20S_%7Bt%2C%5Ctheta%7D%3Df%28%5Csigma_%7Bt%2C%5Ctheta%7D%2C...%2C%5Csigma_%7B0%2C%5Ctheta%7D%29)



It is easy to understand noise Z and state S. But how could we determine what mapping function g.

###### Deep Neural Network

The objective is to approximate the mapping ![img](https://latex.codecogs.com/png.latex?%5Cinline%20Z_%7Bt&plus;1%7D) and ![img](https://latex.codecogs.com/png.latex?%5Cinline%20S_%7Bt%7D) to ![img](https://latex.codecogs.com/png.latex?%5Cinline%20%5Csigma_%7Bt&plus;1%7D) which ideally allows us to generate more data from a given state ![img](https://latex.codecogs.com/png.latex?%5Cinline%20S_%7Bt%7D).Then we could represent this mapping through a deep neural network.

![img](https://latex.codecogs.com/png.latex?%5Cinline%20%5Ctilde%7B%5Csigma%7D_%7Bt&plus;1%2C%5Ctheta%7D%3Dg_%7B%5Ctheta%7D%28Z_%7Bt&plus;1%7D%2C%5Ctilde%7BS%7D_%7Bt%2C%5Ctheta%7D%29)

where

![img](https://latex.codecogs.com/png.latex?%5Cinline%20g%3AL%5E%7B2%7D%28%5Cmathbb%7BR%7D%5E%7BN_%7BZ%7D%7D%29%5Ctimes%20L%5E%7B2%7D%28%5Cmathbb%7BR%7D%5E%7BN_%7BS%7D%7D%29%5Ctimes%20%5CTheta%20%5Crightarrow%20L%5E%7B2%7D%28%5Cmathbb%7BR%7D%5E%7BN_%7BK%7D%5Ctimes%20N_%7BM%7D%7D%29)

![img](https://latex.codecogs.com/png.latex?%5Cinline%20Z_%7Bt&plus;1%7D%5Csim%20N%280%2CI%29)

![img](https://latex.codecogs.com/png.latex?%5Cinline%20%5Ctilde%7BS%7D_%7Bt%2C%5Ctheta%7D%3Df%28%5Ctilde%7B%5Csigma%7D_%7Bt%2C%5Ctheta%7D%2C...%2C%5Ctilde%7B%5Csigma%7D_%7B0%2C%5Ctheta%7D%29)

It is adding ![img](https://latex.codecogs.com/png.latex?%5Cinline%20g_%7B%5Ctheta%7D) as a mapping that relates noise and state to the next time step. ![img](https://latex.codecogs.com/png.latex?%5Cinline%20%5Ctheta) is a nn parameters.

The optimal outcome is to approximate a parameter vector ![img](https://latex.codecogs.com/png.latex?%5Cinline%20%5Ctheta%20_%7BML%7D) inherit the same dynamics in terms of distributional and dependence properties.



##### GANs Model

### **Results and Evaluation**

### Drawback/Challenge

### Future Study
