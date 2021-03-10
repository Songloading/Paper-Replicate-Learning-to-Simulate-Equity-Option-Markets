# Deep Hedging: Learning to Simulate Equity Option Markets

Welcome to our paper replicate project- [Deep Hedging: Learning to Simulate Equity Option Markets](https://arxiv.org/pdf/1911.01700.pdf) by Song Luo and Yunshu Zhang. 

This paper is divided into seven parts; they are:

- Description fo the Problem
- Shortcoming of the Eisting Methods
- Exploratory Data Analysis
- GAN
- Train and Test Model
- Evaluation
- Existing Problem

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

Note: The data is not provided here, if needed, please contact (@Songloading and @jyszhang2020). After put the data into the data/zip folder, then using the below to unpack the zip files into txt file.

```shellscript
#!./src
python ./unzip.py
```

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

All of the above loading data, preprocessing data and computing DLV could be done by running the below script:

```shellscript
#!./src
python ./dlv.py
```

And finally export all the DLV csv file.

```shellscript
#!./src
python ./export.py
```

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



##### Overview of GAN

GAN refers to Generative Adversarial Networks, which has two sets of networks: the generator and the discriminator. The idea is that the generator tries to generate result that seems to be from real domain while the discriminator recieves both the generator-generated result and real data and gives a probability that the given input is from the real domain. The goal probability for discriminator is 0.5: the discriminator cannot tell the input is real or fake. To understand this better, let's take a look at GAN's loss function:

![img](https://latex.codecogs.com/gif.latex?min_%7BG%7Dmax_%7BD%7DV%28D%2CG%29%20%3D%20%5Cmathbb%7BE%7D_%7B_%7Bx%7D%7D%5Blog%28D%28x%29%29%5D%20&plus;%20%5Cmathbb%7BE%7D_%7B_%7Bz%7D%7D%5Blog%281%20-%20D%28G%28z%29%29%29%5D)

where:
- ![img](https://latex.codecogs.com/gif.latex?D%28x%29) is the discriminator's estimate of the probability that real data instance x is from real domain,
- ![img](https://latex.codecogs.com/gif.latex?E_%7Bx%7D) is the expected value over all real data instances,
- ![img](https://latex.codecogs.com/gif.latex?G%28z%29) is the generator-generated result,
- ![img](https://latex.codecogs.com/gif.latex?D%28G%28z%29%29) is the discriminator's estimate of the probability that a fake instance is real, and
- ![img](https://latex.codecogs.com/gif.latex?E_%7Bz%7D) is the expected value over all random inputs to the generator.

The loss functions for both networks are the same except that we want to minimize it for generater and vice versa for discriminator. This is where the term "advsersarial" comes from. GAN has been proven to be effective at generating fake images and image style transfering. In our case, we are going to use GAN to generate simulated DLVs based on the real DLVs, i.e 
![img](https://latex.codecogs.com/png.latex?%5Cinline%20%5Csigma_%7Bt&plus;1%7D%3Dg%28Z_%7Bt&plus;1%7D%2CS_t%29%2Ct%20%5Cin%20%5Cmathbb%7BN%7D_%7B0%7D)

##### Define the architecture
The architecture of GAN we used is very similar to the one from https://github.com/eriklindernoren/PyTorch-GAN. Both the generator and the discriminator recieve the input DLV with size of 3*8 = 24. The last layer of the generator is a tanh activation layer, which we will talk about in the next section while the one for the discriminator is sigmoid, which maps the values from linear layer to the range from 0 to 1.

```python
Generator(
  (model): Sequential(
    (0): Linear(in_features=24, out_features=128, bias=True)
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Linear(in_features=128, out_features=256, bias=True)
    (3): BatchNorm1d(256, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Linear(in_features=256, out_features=512, bias=True)
    (6): BatchNorm1d(512, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace=True)
    (8): Linear(in_features=512, out_features=1024, bias=True)
    (9): BatchNorm1d(1024, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace=True)
    (11): Linear(in_features=1024, out_features=24, bias=True)
    (12): Tanh()
  )
)

Discriminator(
  (model): Sequential(
    (0): Linear(in_features=24, out_features=512, bias=True)
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): LeakyReLU(negative_slope=0.2, inplace=True)
    (4): Linear(in_features=256, out_features=1, bias=True)
    (5): Sigmoid()
  )
)
```

##### Normalize the Data
As for training purpose, we want to normalize the data before fitting to the model. Specific reason can be found here: https://stackoverflow.com/questions/4674623/why-do-we-have-to-normalize-the-input-for-an-artificial-neural-network. However, our data is not normaly distributed, which can be seen from the histogram below, and thus we use Z-score normalization. 
![img](https://drive.google.com/uc?export=view&id=1N6l-pm-nzNhLXNue5T_GqE3sn2TkcHq9)

As mentioned in the previous section, we use tanh as our last layer for the generator. This is because as we normalize the input from -1 to 1, we will also want the output from the model in this range. The max&min normalization is used for this purpose. The normalization functions we used are:
```python
def NormalizeData(data):
    return 2*(data - np.min(data)) / (np.max(data) - np.min(data))-1

def Z_Score_NormalizeData(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean)/std
```

### Train and Test the model
- Train a model:
```shellscript
#!./src
python ./Project.py --mode=Train --num_epochs=np --batch_size=bs --learning_rate=lr --save_model=sm
```
where you should replace np, bs, lr, and sm by your choice. e.g:
```shellscript
#!./src
python ./Project.py --mode=Train --num_epochs=100 --batch_size=3 --learning_rate=3e-4 --save_model=True
```
If the save_model argument is true, the parameters of your generator should be saved to the model folder. We strongly recommend to set the batch size less than 5 because of the limit of data.

- Test a model:
```shellscript
#!./src
python ./Project.py --mode=Test --dlv_path=dp --recursive=r --recursive_length=rl
```
where dp, r, and rl are the path to the starting DLV, if you want to test the model recursively (other wise it only generate one single result), and how many steps you want to test the model. e.g:
```shellscript
#!./src
python ./Project.py --mode=Test --dlv_path='data/spxw_call_dlv_0.csv' --recursive=True --recursive_length=10
```
This will test the model starting with DLV of Jan 1st, 2020 and you should expect there are 10 results generated, which represent the simulated DLVs from Jan 2nd, 2020 to Jan 11th, 2020.


### **Evaluation**
The performance of the model is evaluated by three metrics: CEPDF (Cumulative Empirical Probability Density Function), Skew, and Kurtosis.
- CEPDF:
   Since we want our simulated results to match the real data, the Empirical Probability Density Function can be used for examing the distributional properties. Due to the 	random extreme values in the real dataset, we bining the time series data such that each bin contains only 5 DLVs. Then calculate the EPDF for both simulated and real 	 bins. The result is calculated by suming the differences between each bin.
   Let ![img](https://latex.codecogs.com/gif.latex?%5Cbeta%20_%7Bh%7D%20%3D%20%5B%7B%5Cbeta%20_%7B1%7D%2C...%5Cbeta%20_%7BK%7D%7D%5D) be the bins
   Thus we can find the empirical probability density function for each bin:
   ![img](https://latex.codecogs.com/gif.latex?%5Ctilde%7Bf_%7Bh%7D%7D%3A%5Cbeta%20_%7Bh%7D%20%5Crightarrow%20%5Cmathbb%7BR%7D_%7B%5Cgeq%200%7D)
   
   ![img](https://latex.codecogs.com/gif.latex?%5Ctilde%7Bf_%7Bg%7D%7D%3A%5Cbeta%20_%7Bh%7D%20%5Crightarrow%20%5Cmathbb%7BR%7D_%7B%5Cgeq%200%7D)
   
   where ![img](https://latex.codecogs.com/gif.latex?%5Ctilde%7Bf_%7Bh%7D%7D) and ![img](https://latex.codecogs.com/gif.latex?%5Ctilde%7Bf_%7Bg%7D%7D) refer to the epdf for real  and simulated bins
   
   Then we can calculate the cumulative difference:
   ![img](https://latex.codecogs.com/gif.latex?%5Csum_%7BB%5Cin%5Cbeta%20_%7Bh%7D%7D%5Cleft%20%7C%20%5Ctilde%7Bf%7D_%7Bh%7D%20-%20%5Ctilde%7Bf%7D_%7Bg%7D%20%5Cright%20%7C)
	
- Skew & Kurtosis:
  Skew and Kurtosis are measurements of the asymmetry and the flatness of the probability distribution. In financial applications higher order moments such as the skewness and kurtosis are of interest as they determine the propensity to generate extremal values.
  For calculation, we simply find the Skew and Kurtosis for both the simulated and the real distributions, and subtract them:
  
  ![img](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7BN%7D%5Csum%20%5Cleft%20%5C%7C%20Skew%28a_%7Bh%7D%29%20-%20Skew%28a_%7Bg%7D%29%20%5Cright%20%5C%7C)
  
  and similarly:
  
  ![img](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7BN%7D%5Csum%20%5Cleft%20%5C%7C%20Kurtosis%28a_%7Bh%7D%29%20-%20Kurtosis%28a_%7Bg%7D%29%20%5Cright%20%5C%7C)
  
  where:
  ![img](https://latex.codecogs.com/gif.latex?a_%7Bh%7D) and ![img](https://latex.codecogs.com/gif.latex?a_%7Bg%7D) are the real and simulated DLVs.

- Evaluate:
```shellscript
#!./src
python ./metrics.py
```
Note: Please ensure that the number of files in **result** folder is greater than 10 and also please include only the csv files you want to test in the **result** folder.
### Existing Problem

- Hardware issue

  Since all of the option data file are pretty large it is hard to loading them or mauniplate them at the same time. It is better to have a virtual machine to handle it or Google Colab.

- Lack of data

  Right now we are using the data from five months from 2020, which is definitely not enough to run a deep learning model, compare to the original paper that using nine years' data. 

  We find that we could acquire more data from the [Wikitter](https://www.wikitter.com/). Hence lacking of data should not be a huge deal in the furture study.

- 2020 stock market crash

  2020 is not a good year in terms of the stock market. As we all know, the global stock market crash began on 20 February 2020 and ended on 7 April. And we are using February and March data for training which will cause bias.

  We think that maybe GANs is not robust enough to handle such case.  There will be a lot of room for us to develop a more powerful model.

- Computing DLV issue

  As we showed above, computing DLV requires a lot of work and financial mathematical knowledge. If it is possible, we should get in touch with JP Morgan computer scientist in order to ensure we are calculating it in the correct way.

