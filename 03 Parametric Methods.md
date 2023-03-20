# Parametric Methods

### There is a set of parameters that determine a probability model.


```python
import numpy as np
np.random.seed(1)
N = 100
alpha = 2.5
beta = 0.9
eps = np.random.normal(0, 0.5, size=N)
oriX = np.random.normal(10, 1, N)
realY = alpha + beta * oriX
oriY = realY + eps
import arviz as az
az.plot_kde(oriY)
```




    <AxesSubplot:>




    
![png](output_2_1.png)
    



```python
import matplotlib.pyplot as plt

plt.plot(oriX, oriY, 'C0.')

plt.plot(oriX, realY, 'k')
plt.show()
```


    
![png](output_3_0.png)
    


    WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.



```python
import pymc3 as pm

with pm.Model() as gaussian:
    a = pm.Normal('a', mu=0, sd=10)
    b = pm.Normal('b', mu=0, sd=1)
    e = pm.HalfCauchy('e', 5)
    
    u = pm.Deterministic('u', a + b * oriX)
    yPred = pm.Normal('yPred', mu=u, sd=e, observed=oriY)
    
    trace_g = pm.sample(2000, tune=1000)
```

    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:11: FutureWarning: In v4.0, pm.sample will return an `arviz.InferenceData` object instead of a `MultiTrace` by default. You can pass return_inferencedata=True or return_inferencedata=False to be safe and silence this warning.
      # This is added back by InteractiveShellApp.init_path()
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [e, b, a]




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
  <progress value='12000' class='' max='12000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [12000/12000 00:14<00:00 Sampling 4 chains, 2 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 2_000 draw iterations (4_000 + 8_000 draws total) took 30 seconds.
    There were 2 divergences after tuning. Increase `target_accept` or reparameterize.



```python
az.plot_trace(trace_g, var_names=['a', 'b', 'e'], compact=False)
plt.figure()
```

    Got error No model on context stack. trying to find log_likelihood in translation.
    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/arviz/data/io_pymc3_3x.py:102: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.
      FutureWarning,
    Got error No model on context stack. trying to find log_likelihood in translation.





    <Figure size 432x288 with 0 Axes>




    
![png](output_5_2.png)
    



    <Figure size 432x288 with 0 Axes>


## Regression


```python
np.random.seed(1)

n = 30
X0 = np.sort(3 * np.random.rand(n))[:, None]

m = 100
X = np.linspace(0, 3, m)[:, None] 

noise = 0.1
lengthscale = 0.3

cov = pm.gp.cov.ExpQuad(1,  lengthscale)
K = cov(X0)
K_s = cov(X0, X)
K_noise = K + noise * np.eye(n)

K_stable = K + 1e-12 * np.eye(n)
```


```python
f = np.random.multivariate_normal(mean=np.zeros(n), cov=K_noise.eval())

fig, ax = plt.subplots(figsize=(16,4))
ax.scatter(X0, f, s=40, color='b', label='True points')

ax.set_xlim(0,3)
ax.set_ylim(-2,2)
ax.legend()
```




    <matplotlib.legend.Legend at 0x7f7cd3c764d0>




    
![png](output_8_1.png)
    



```python
L = np.linalg.cholesky(K_noise.eval())
alpha = np.linalg.solve(L.T, np.linalg.solve(L, f))
post_mean = np.dot(K_s.T.eval(), alpha)

ax.plot(X, post_mean, color='g', alpha=0.8, label='Posterior mean')

ax.set_xlim(0,3)
ax.set_ylim(-2,2)
ax.legend()
```




    <matplotlib.legend.Legend at 0x7fd9f9360390>




    
![png](output_9_1.png)
    



```python
import theano.tensor as tt

with pm.Model() as model:
    f_sample = pm.Flat('f_sample', shape=(n,))
    
    y = pm.MvNormal('y', observed=f, mu=f_sample, cov = noise* tt.eye(n), shape=n)
    
    L = tt.slinalg.cholesky(K_noise)
    f_pred = pm.Deterministic('f_pred', tt.dot(K_s.T, tt.slinalg.solve(L.T, tt.slinalg.solve(L, f_sample))))
    
    ess_step = pm.EllipticalSlice(vars=[f_sample], prior_cov=K_stable)
    trace = pm.sample(5000, start=model.test_point, step=[ess_step], progressbar=False, random_seed=1)
```

    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:12: FutureWarning: In v4.0, pm.sample will return an `arviz.InferenceData` object instead of a `MultiTrace` by default. You can pass return_inferencedata=True or return_inferencedata=False to be safe and silence this warning.
      if sys.path[0] == '':
    Multiprocess sampling (4 chains in 4 jobs)
    EllipticalSlice: [f_sample]
    Sampling 4 chains for 1_000 tune and 5_000 draw iterations (4_000 + 20_000 draws total) took 30 seconds.
    The estimated number of effective samples is smaller than 200 for some parameters.



```python
fig, ax = plt.subplots(figsize=(16,4))
for idx in np.random.randint(4000, 5000, 500):
    ax.plot(X, trace['f_pred'][idx], alpha=0.02, color='navy')
    
ax.scatter(X0, f, s=40, color='k', label='True points')
ax.plot(X, post_mean, color='g', alpha=0.8, label='Posterior mean')
ax.set_xlim(0,3)
ax.set_ylim(-2,2)
ax.legend()
```




    <matplotlib.legend.Legend at 0x7fd9f63c9050>




    
![png](output_11_1.png)
    


## Classification


```python
np.random.seed(5)
f = np.random.multivariate_normal(mean=np.zeros(n), cov=K_stable.eval())

f[f > 0] = 1
f[f <= 0] = 0

print(f)
```

    [1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1.]



```python
fig, ax = plt.subplots(figsize=(16,4))
ax.scatter(X0, np.ma.masked_where(f==0, f), color='b', label = 'negative')
ax.scatter(X0, np.ma.masked_where(f==1, f), color='r', label = 'positive')

ax.legend(loc='lower right')
ax.set_xlim(-0.1,3.1)
ax.set_ylim(-0.2,1.2)
```




    (-0.2, 1.2)




    
![png](output_14_1.png)
    



```python
with pm.Model() as binary:
    f_sample = pm.Flat('f_sample', shape=n)
    f_transform = pm.invlogit(f_sample)
    
    y = pm.Binomial('y', observed=f, n=np.ones(n), p=f_transform, shape=n)
    
    f_pred = pm.Deterministic('f_pred', tt.dot(K.T, tt.slinalg.solve(L.T, tt.slinalg.solve(L, f_transform))))

    Y_obs = pm.Bernoulli("Y_obs", f_pred, observed=f)
    
    # A different trace was created, called traceB (for Binary).
    traceB = pm.sample(5000, return_inferencedata=False, chains=4)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [f_sample]




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
  <progress value='24000' class='' max='24000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [24000/24000 01:06<00:00 Sampling 4 chains, 19,985 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 5_000 draw iterations (4_000 + 20_000 draws total) took 82 seconds.
    There were 4996 divergences after tuning. Increase `target_accept` or reparameterize.
    The chain contains only diverging samples. The model is probably misspecified.
    The acceptance probability does not match the target. It is 0.6982323456911356, but should be close to 0.8. Try to increase the number of tuning steps.
    The chain contains only diverging samples. The model is probably misspecified.
    The acceptance probability does not match the target. It is 0.9556740863442501, but should be close to 0.8. Try to increase the number of tuning steps.
    There were 4989 divergences after tuning. Increase `target_accept` or reparameterize.
    The rhat statistic is larger than 1.4 for some parameters. The sampler did not converge.
    The estimated number of effective samples is smaller than 200 for some parameters.



```python
fig, ax = plt.subplots(figsize=(16,4))
for idx in np.random.randint(4000, 5000, 500):
    ax.plot(X0, traceB['f_pred'][idx], alpha=0.02, color='navy')
    
ax.scatter(X0, f, s=40, color='k', label='True points')
ax.legend(loc='lower right')
ax.set_xlim(0,3)
ax.set_ylim(-0.1,1.1)
ax.legend()
```




    <matplotlib.legend.Legend at 0x7fd841da8690>




    
![png](output_16_1.png)
    


Recommendation: https://www.jmlr.org/papers/volume9/nickisch08a/nickisch08a.pdf

### Back from trace


```python
plt.figure()

plt.plot(oriX, oriY, 'C0.')

alpha_m = trace_g['a'].mean()
beta_m = trace_g['b'].mean()

draws = range(0, len(trace_g['a']), 10)
              
plt.plot(oriX, trace_g['a'][draws]+trace_g['b'][draws]*oriX[:, np.newaxis], c='lightblue', alpha=0.5)
plt.plot(oriX, alpha_m+beta_m*oriX, c='teal', label=f'y = {alpha_m: .2f} + {beta_m: .2f} * X')
              
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fabf45d6710>




    
![png](output_19_1.png)
    



```python
ppc = pm.sample_posterior_predictive(trace_g, samples=2000, model=gaussian)

az.plot_hdi(oriX, ppc['yPred'], hdi_prob=0.1, color='lightblue')

plt.plot(oriX, oriY, 'b.')
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
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:02<00:00]
</div>






    [<matplotlib.lines.Line2D at 0x7fabf41baed0>]




    
![png](output_20_2.png)
    



```python
trace_g.varnames
```




    ['a', 'b', 'e_log__', 'e', 'u']


