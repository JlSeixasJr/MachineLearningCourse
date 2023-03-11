# Bayesian Methods

$$P(A|B)=\frac{P(B|A).P(A)}{P(B)}$$

* A,B -> Events
* P(A|B) -> Probability of A given B.
* P(B|A) -> Probability of B given A.
* P(A),P(B) -> Independent probabilities.

https://www.youtube.com/watch?v=R13BD8qKeTg


```python
import warnings
warnings.simplefilter(action='ignore')
```


```python
import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv")
data = data.sample(frac=0.01, random_state=1)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>insert_date</th>
      <th>origin</th>
      <th>destination</th>
      <th>start_date</th>
      <th>end_date</th>
      <th>train_type</th>
      <th>price</th>
      <th>train_class</th>
      <th>fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15248</th>
      <td>2019-04-26 01:00:46</td>
      <td>MADRID</td>
      <td>SEVILLA</td>
      <td>2019-05-01 15:00:00</td>
      <td>2019-05-01 17:21:00</td>
      <td>AVE</td>
      <td>NaN</td>
      <td>Preferente</td>
      <td>Flexible</td>
    </tr>
    <tr>
      <th>14882</th>
      <td>2019-04-20 07:35:05</td>
      <td>SEVILLA</td>
      <td>MADRID</td>
      <td>2019-05-01 21:00:00</td>
      <td>2019-05-01 23:52:00</td>
      <td>AV City</td>
      <td>49.15</td>
      <td>Turista</td>
      <td>Promo</td>
    </tr>
    <tr>
      <th>23985</th>
      <td>2019-04-14 07:03:58</td>
      <td>MADRID</td>
      <td>VALENCIA</td>
      <td>2019-05-21 07:40:00</td>
      <td>2019-05-21 09:20:00</td>
      <td>AVE</td>
      <td>57.75</td>
      <td>Turista</td>
      <td>Promo</td>
    </tr>
    <tr>
      <th>22378</th>
      <td>2019-04-22 20:02:09</td>
      <td>MADRID</td>
      <td>SEVILLA</td>
      <td>2019-04-25 21:25:00</td>
      <td>2019-04-26 00:10:00</td>
      <td>AV City</td>
      <td>49.15</td>
      <td>Turista</td>
      <td>Promo</td>
    </tr>
    <tr>
      <th>1901</th>
      <td>2019-04-11 23:33:00</td>
      <td>MADRID</td>
      <td>VALENCIA</td>
      <td>2019-06-04 16:05:00</td>
      <td>2019-06-04 22:47:00</td>
      <td>REGIONAL</td>
      <td>28.35</td>
      <td>Turista</td>
      <td>Adulto ida</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.isnull().sum()/len(data)
```




    insert_date    0.000000
    origin         0.000000
    destination    0.000000
    start_date     0.000000
    end_date       0.000000
    train_type     0.000000
    price          0.116279
    train_class    0.003876
    fare           0.003876
    dtype: float64




```python
data['train_class'] = data['train_class'].fillna(data['train_class'].mode().iloc[0])
data['fare'] = data['fare'].fillna(data['fare'].mode().iloc[0])
data['price'] = data.groupby('fare').transform(lambda x: x.fillna(x.mean()))

data.isnull().sum()/len(data)
```




    insert_date    0.0
    origin         0.0
    destination    0.0
    start_date     0.0
    end_date       0.0
    train_type     0.0
    price          0.0
    train_class    0.0
    fare           0.0
    dtype: float64



## Gaussian inferences

<img src="continuous.png">


```python
import arviz as az

import matplotlib.pyplot as plt
%matplotlib inline

az.plot_kde(data['price'].values, rug=True)
plt.yticks([0], alpha=0)
```




    ([<matplotlib.axis.YTick at 0x7fd3906ca690>], [Text(0, 0, '')])




    
![png](output_10_1.png)
    



```python
import pymc3 as pm

print('Running on PyMC3 v{}'.format(pm.__version__))

with pm.Model() as model:
    mu = pm.Uniform('mu', lower=0, upper=300) #Use any prior knowledge 
    sigma = pm.HalfNormal('sigma', sd=10)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data['price'].values)
    
    trace = pm.sample(1000, tune=1000)
```

    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.


    Running on PyMC3 v3.11.4


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [sigma, mu]
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 04:15<00:00 Sampling 4 chains, 0 divergences]
</div>



    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/tensor/elemwise.py:826: RuntimeWarning: overflow encountered in exp
      variables = ufunc(*ufunc_args, **ufunc_kwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:2893: RuntimeWarning: divide by zero encountered in log
      return np.log(x)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/tensor/elemwise.py:826: RuntimeWarning: divide by zero encountered in impl (vectorized)
      variables = ufunc(*ufunc_args, **ufunc_kwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:1955: RuntimeWarning: invalid value encountered in true_divide
      return x / y
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:2893: RuntimeWarning: divide by zero encountered in log
      return np.log(x)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/tensor/elemwise.py:826: RuntimeWarning: divide by zero encountered in impl (vectorized)
      variables = ufunc(*ufunc_args, **ufunc_kwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:87: RuntimeWarning: overflow encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:1955: RuntimeWarning: invalid value encountered in true_divide
      return x / y
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:87: RuntimeWarning: overflow encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/tensor/elemwise.py:826: RuntimeWarning: overflow encountered in exp
      variables = ufunc(*ufunc_args, **ufunc_kwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:2893: RuntimeWarning: divide by zero encountered in log
      return np.log(x)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/tensor/elemwise.py:826: RuntimeWarning: divide by zero encountered in impl (vectorized)
      variables = ufunc(*ufunc_args, **ufunc_kwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:1955: RuntimeWarning: invalid value encountered in true_divide
      return x / y
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:2893: RuntimeWarning: divide by zero encountered in log
      return np.log(x)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/tensor/elemwise.py:826: RuntimeWarning: divide by zero encountered in impl (vectorized)
      variables = ufunc(*ufunc_args, **ufunc_kwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:87: RuntimeWarning: overflow encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:1955: RuntimeWarning: invalid value encountered in true_divide
      return x / y
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/tensor/elemwise.py:826: RuntimeWarning: overflow encountered in exp
      variables = ufunc(*ufunc_args, **ufunc_kwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 272 seconds.



```python
with model:
    print(trace)
```

    <MultiTrace: 4 chains, 1000 iterations, 4 variables>



```python
trace.varnames
```




    ['mu_interval__', 'sigma_log__', 'mu', 'sigma']




```python
az.plot_trace(trace, compact=False)
```

    Got error No model on context stack. trying to find log_likelihood in translation.
    Got error No model on context stack. trying to find log_likelihood in translation.





    array([[<AxesSubplot:title={'center':'mu'}>,
            <AxesSubplot:title={'center':'mu'}>],
           [<AxesSubplot:title={'center':'sigma'}>,
            <AxesSubplot:title={'center':'sigma'}>]], dtype=object)




    
![png](output_14_2.png)
    



```python
az.plot_joint(trace, kind='kde', fill_last=False)
```

    Got error No model on context stack. trying to find log_likelihood in translation.





    array([<AxesSubplot:xlabel='mu', ylabel='sigma'>, <AxesSubplot:>,
           <AxesSubplot:>], dtype=object)




    
![png](output_15_2.png)
    



```python
az.summary(trace)
```

    Got error No model on context stack. trying to find log_likelihood in translation.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu</th>
      <td>62.004</td>
      <td>1.501</td>
      <td>59.203</td>
      <td>64.716</td>
      <td>0.026</td>
      <td>0.018</td>
      <td>3349.0</td>
      <td>2539.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>23.939</td>
      <td>1.050</td>
      <td>22.032</td>
      <td>25.922</td>
      <td>0.018</td>
      <td>0.012</td>
      <td>3622.0</td>
      <td>2915.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_posterior(trace)
```

    Got error No model on context stack. trying to find log_likelihood in translation.





    array([<AxesSubplot:title={'center':'mu'}>,
           <AxesSubplot:title={'center':'sigma'}>], dtype=object)




    
![png](output_17_2.png)
    



```python
pm.plots.autocorrplot(trace, figsize=(10,5))
```

    Got error No model on context stack. trying to find log_likelihood in translation.





    array([[<AxesSubplot:title={'center':'mu\n0'}>,
            <AxesSubplot:title={'center':'mu\n1'}>,
            <AxesSubplot:title={'center':'mu\n2'}>,
            <AxesSubplot:title={'center':'mu\n3'}>],
           [<AxesSubplot:title={'center':'sigma\n0'}>,
            <AxesSubplot:title={'center':'sigma\n1'}>,
            <AxesSubplot:title={'center':'sigma\n2'}>,
            <AxesSubplot:title={'center':'sigma\n3'}>]], dtype=object)




    
![png](output_18_2.png)
    



```python
bfmi = pm.bfmi(trace)
pm.energyplot(trace, figsize=(6,4))
```

    Got error No model on context stack. trying to find log_likelihood in translation.
    Got error No model on context stack. trying to find log_likelihood in translation.





    <AxesSubplot:>




    
![png](output_19_2.png)
    


##### Bayesian Fraction of Missing Information
BFMI quantifies how well momentum resampling matches the marginal energy distribution.

## Posterior Predictive Checks


```python
import numpy as np

ppc = pm.sample_posterior_predictive(trace, samples=1000, model=model)

np.asarray(ppc['y']).shape
```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='1000' class='' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1000/1000 00:01<00:00]
</div>






    (1000, 258)




```python
_, ax = plt.subplots(figsize=(10,5))
ax.hist([y.mean() for y in ppc['y']], bins=19, alpha=0.5)
ax.axvline(data.price.mean())
ax.set(title='PPC')
```




    [Text(0.5, 1.0, 'PPC')]




    
![png](output_23_1.png)
    



```python
az.plot_forest(trace, combined=True)
```

    Got error No model on context stack. trying to find log_likelihood in translation.





    array([<AxesSubplot:title={'center':'94.0% HDI'}>], dtype=object)




    
![png](output_24_2.png)
    


## Group Comparison


```python
import seaborn as sns
sns.boxplot(x='fare', y='price', data=data)
```




    <AxesSubplot:xlabel='fare', ylabel='price'>




    
![png](output_26_1.png)
    



```python
categories = data.fare.unique()
categories
```




    array(['Flexible', 'Promo', 'Adulto ida', 'Promo +'], dtype=object)



## Estimate per Groups


```python
price = data['price'].values
idx = pd.Categorical(data['fare'], categories=categories).codes
groups = len(np.unique(idx))
```


```python
with pm.Model() as compGroups:
    mu = pm.Normal('mu', mu=0, sd=10, shape=groups) 
    sigma = pm.HalfNormal('sigma', sd=10, shape=groups)
    y = pm.Normal('y', mu=mu[idx], sigma=sigma[idx], observed=price)
    
    trace_g = pm.sample(5000, tune=5000)

az.plot_trace(trace_g, compact=False)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [sigma, mu]
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='40000' class='' max='40000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [40000/40000 1:48:53<00:00 Sampling 4 chains, 934 divergences]
</div>



    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:87: RuntimeWarning: overflow encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:1955: RuntimeWarning: invalid value encountered in true_divide
      return x / y
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:2893: RuntimeWarning: divide by zero encountered in log
      return np.log(x)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:87: RuntimeWarning: overflow encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:1955: RuntimeWarning: invalid value encountered in true_divide
      return x / y
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:2893: RuntimeWarning: divide by zero encountered in log
      return np.log(x)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:87: RuntimeWarning: overflow encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/tensor/elemwise.py:826: RuntimeWarning: overflow encountered in exp
      variables = ufunc(*ufunc_args, **ufunc_kwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:1955: RuntimeWarning: invalid value encountered in true_divide
      return x / y
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:2893: RuntimeWarning: divide by zero encountered in log
      return np.log(x)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:87: RuntimeWarning: overflow encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/tensor/elemwise.py:826: RuntimeWarning: overflow encountered in exp
      variables = ufunc(*ufunc_args, **ufunc_kwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    Sampling 4 chains for 5_000 tune and 5_000 draw iterations (20_000 + 20_000 draws total) took 6549 seconds.
    There were 21 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 272 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 32 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 609 divergences after tuning. Increase `target_accept` or reparameterize.
    Got error No model on context stack. trying to find log_likelihood in translation.
    Got error No model on context stack. trying to find log_likelihood in translation.





    array([[<AxesSubplot:title={'center':'mu\n0'}>,
            <AxesSubplot:title={'center':'mu\n0'}>],
           [<AxesSubplot:title={'center':'mu\n1'}>,
            <AxesSubplot:title={'center':'mu\n1'}>],
           [<AxesSubplot:title={'center':'mu\n2'}>,
            <AxesSubplot:title={'center':'mu\n2'}>],
           [<AxesSubplot:title={'center':'mu\n3'}>,
            <AxesSubplot:title={'center':'mu\n3'}>],
           [<AxesSubplot:title={'center':'sigma\n0'}>,
            <AxesSubplot:title={'center':'sigma\n0'}>],
           [<AxesSubplot:title={'center':'sigma\n1'}>,
            <AxesSubplot:title={'center':'sigma\n1'}>],
           [<AxesSubplot:title={'center':'sigma\n2'}>,
            <AxesSubplot:title={'center':'sigma\n2'}>],
           [<AxesSubplot:title={'center':'sigma\n3'}>,
            <AxesSubplot:title={'center':'sigma\n3'}>]], dtype=object)




    
![png](output_30_4.png)
    



```python
flat_fares = az.from_pymc3(trace=trace_g)
fares_gaussian = az.summary(flat_fares)
fares_gaussian
```

    Got error No model on context stack. trying to find log_likelihood in translation.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu[0]</th>
      <td>72.242</td>
      <td>3.181</td>
      <td>66.439</td>
      <td>78.292</td>
      <td>0.030</td>
      <td>0.021</td>
      <td>11146.0</td>
      <td>10929.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu[1]</th>
      <td>58.859</td>
      <td>1.584</td>
      <td>55.880</td>
      <td>61.831</td>
      <td>0.012</td>
      <td>0.009</td>
      <td>16192.0</td>
      <td>10295.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu[2]</th>
      <td>31.481</td>
      <td>1.581</td>
      <td>28.435</td>
      <td>34.436</td>
      <td>0.013</td>
      <td>0.010</td>
      <td>14238.0</td>
      <td>9719.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu[3]</th>
      <td>28.894</td>
      <td>4.572</td>
      <td>19.603</td>
      <td>35.436</td>
      <td>0.066</td>
      <td>0.046</td>
      <td>7416.0</td>
      <td>5439.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma[0]</th>
      <td>23.500</td>
      <td>2.210</td>
      <td>19.441</td>
      <td>27.677</td>
      <td>0.019</td>
      <td>0.014</td>
      <td>13539.0</td>
      <td>11246.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma[1]</th>
      <td>21.133</td>
      <td>1.119</td>
      <td>19.110</td>
      <td>23.293</td>
      <td>0.009</td>
      <td>0.007</td>
      <td>14478.0</td>
      <td>10287.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma[2]</th>
      <td>6.973</td>
      <td>1.231</td>
      <td>4.899</td>
      <td>9.282</td>
      <td>0.010</td>
      <td>0.008</td>
      <td>15060.0</td>
      <td>11762.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma[3]</th>
      <td>5.874</td>
      <td>3.995</td>
      <td>1.370</td>
      <td>13.618</td>
      <td>0.055</td>
      <td>0.039</td>
      <td>5970.0</td>
      <td>7621.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from scipy import stats
dist = stats.norm()

_, ax = plt.subplots(3, 2, figsize=(20, 12), constrained_layout=True)

comparisons = [(i, j) for i in range(4) for j in range(i+1, 4)]
pos = [(k, l) for k in range(5) for l in (0, 1)]

for (i, j), (k, l) in zip(comparisons, pos):
    means_diff = trace_g['mu'][:, i] - trace_g['mu'][:, j]
    d_cohen = (means_diff / np.sqrt((trace_g['sigma'][:, i]**2 + trace_g['sigma'][:, j]**2) / 2)).mean()
    ps = dist.cdf(d_cohen/(2**0.5))
    az.plot_posterior(means_diff, ref_val=0, ax=ax[k, l])
    ax[k, l].set_title(f'$\mu_{i}-\mu_{j}$')
    ax[k, l].plot(
        0, label=f"Cohen's d = {d_cohen:.2f}\nProb sup = {ps:.2f}", alpha=0)
    ax[k, l].legend();
```


    
![png](output_32_0.png)
    


### Simpler Model


```python
np.random.seed(23)
alpha = -0.8
beta = [-1, 1.5]
sigma = 1
size = 500
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.8
Y = alpha + beta[0] * 0.8 + beta[1] * X2 + np.random.randn(size) * sigma
df = pd.DataFrame(
    data = np.array([X1, X2, Y]),
    index = ['X1', 'X2', 'Y']
).T
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.666988</td>
      <td>-1.366156</td>
      <td>-2.097039</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.025813</td>
      <td>0.327330</td>
      <td>-2.586322</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.777619</td>
      <td>-0.411806</td>
      <td>-2.417593</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.948634</td>
      <td>-0.414930</td>
      <td>-3.512784</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.701672</td>
      <td>-0.652830</td>
      <td>-5.343569</td>
    </tr>
  </tbody>
</table>
</div>




```python
with pm.Model() as model1:
    X_1 = pm.Data('X1', X1)
    X_2 = pm.Data('X2', X2)
    
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=10)
    
    mu = alpha + beta[0] * X_1 + beta[1] * X_2
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)
    
    trace = pm.sample(100, return_inferencedata=False, chains=4)
    
az.plot_trace(trace)
```

    Only 100 samples in chain.
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [sigma, beta, alpha]
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='4400' class='' max='4400' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [4400/4400 12:42<00:00 Sampling 4 chains, 0 divergences]
</div>



    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    Sampling 4 chains for 1_000 tune and 100 draw iterations (4_000 + 400 draws total) took 780 seconds.
    Got error No model on context stack. trying to find log_likelihood in translation.
    Got error No model on context stack. trying to find log_likelihood in translation.





    array([[<AxesSubplot:title={'center':'alpha'}>,
            <AxesSubplot:title={'center':'alpha'}>],
           [<AxesSubplot:title={'center':'beta'}>,
            <AxesSubplot:title={'center':'beta'}>],
           [<AxesSubplot:title={'center':'sigma'}>,
            <AxesSubplot:title={'center':'sigma'}>]], dtype=object)




    
![png](output_35_4.png)
    



```python
with model1:
    pm.set_data({
        'X1': [0.6, 0.02],
        'X2': [-1.3, 0.3]
    })
    y_test = pm.fast_sample_posterior_predictive(trace, samples=100)
print(y_test['Y_obs'].mean(axis=0))
```

    [-3.40412464 -1.27126689]


### Generalized Linear Models


```python
from pymc3.glm import GLM

with pm.Model() as model_glm:
    GLM.from_formula('Y ~ X1 + X2', df)
    trace = pm.sample(100)
    pm.traceplot(trace)
    plt.show()
```

    The glm module is deprecated and will be removed in version 4.0
    We recommend to instead use Bambi https://bambinos.github.io/bambi/
    Only 100 samples in chain.
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [sd, X2, X1, Intercept]
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='4400' class='' max='4400' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [4400/4400 04:13<00:00 Sampling 4 chains, 0 divergences]
</div>



    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:3167: RuntimeWarning: overflow encountered in double_scalars
      return x * x
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/tensor/elemwise.py:826: RuntimeWarning: overflow encountered in impl (vectorized)
      variables = ufunc(*ufunc_args, **ufunc_kwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:2893: RuntimeWarning: divide by zero encountered in log
      return np.log(x)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/tensor/elemwise.py:826: RuntimeWarning: divide by zero encountered in impl (vectorized)
      variables = ufunc(*ufunc_args, **ufunc_kwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:87: RuntimeWarning: overflow encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/theano/scalar/basic.py:1955: RuntimeWarning: invalid value encountered in true_divide
      return x / y
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    Sampling 4 chains for 1_000 tune and 100 draw iterations (4_000 + 400 draws total) took 270 seconds.



    
![png](output_38_3.png)
    



```python
df['Y_bin'] = df['Y']>df['Y'].mean()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>Y</th>
      <th>Y_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.666988</td>
      <td>-1.366156</td>
      <td>-2.097039</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.025813</td>
      <td>0.327330</td>
      <td>-2.586322</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.777619</td>
      <td>-0.411806</td>
      <td>-2.417593</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.948634</td>
      <td>-0.414930</td>
      <td>-3.512784</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.701672</td>
      <td>-0.652830</td>
      <td>-5.343569</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
with pm.Model() as model_log:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    
    p = pm.Deterministic('p', pm.math.sigmoid(alpha+beta[0]*X1+beta[1]*X2))
    
    Y_obs = pm.Bernoulli("Y_obs", p, observed=df['Y_bin'])
    
    traceB = pm.sample(100, return_inferencedata=False, chains=4) 
    
az.plot_trace(traceB)
```

    Only 100 samples in chain.
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [beta, alpha]
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='4400' class='' max='4400' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [4400/4400 04:46<00:00 Sampling 4 chains, 0 divergences]
</div>



    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    Sampling 4 chains for 1_000 tune and 100 draw iterations (4_000 + 400 draws total) took 303 seconds.
    Got error No model on context stack. trying to find log_likelihood in translation.
    Got error No model on context stack. trying to find log_likelihood in translation.





    array([[<AxesSubplot:title={'center':'alpha'}>,
            <AxesSubplot:title={'center':'alpha'}>],
           [<AxesSubplot:title={'center':'beta'}>,
            <AxesSubplot:title={'center':'beta'}>],
           [<AxesSubplot:title={'center':'p'}>,
            <AxesSubplot:title={'center':'p'}>]], dtype=object)




    
![png](output_40_4.png)
    



```python
from pymc3.glm.families import Binomial
with pm.Model() as model_glm_logistic:
    GLM.from_formula('Y_bin ~ X1 + X2', df, family=Binomial())
    trace = pm.sample(100)
    pm.traceplot(trace)
    plt.show()
```

    The glm module is deprecated and will be removed in version 4.0
    We recommend to instead use Bambi https://bambinos.github.io/bambi/
    Only 100 samples in chain.
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [X2, X1, Intercept]
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='4400' class='' max='4400' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [4400/4400 04:54<00:00 Sampling 4 chains, 0 divergences]
</div>



    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    Sampling 4 chains for 1_000 tune and 100 draw iterations (4_000 + 400 draws total) took 312 seconds.



    
![png](output_41_3.png)
    


http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf
