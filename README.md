# Machine Learning Practical

## Personal Information

#### José Luis Seixas Junior
#### email: jlseixasjr@inf.elte.hu
#### Room: 7.33


| Reference Week | Practical |
| :-: | :- | 
| Feb 27 | Before classes |
| Mar 06 | Introduction |
| Mar 13 | Bayesian Methods |
| Mar 20 | Parametric Methods |
| Mar 27 | PGM |
| Apr 03 | HMM |
| Apr 10 | **Easter Monday** |
| Apr 17 | Markov Chain + LDA |
| Apr 24 | Lazy Learner |
| May 01 | **Labor day** |
| May 08 | Decision Tree |
| May 15 | Kernel Machines |
| May 22 | Consultation |

## Packages
* numpy
* matplotlib
* pandas
* sklearn
* hmmlearn
* arviz
* pymc3
* pgmpy
* theano
* gensim
* nltk

### Optionals
* optunity
* tqdm
* seaborn
* mlxtend

```python
import warnings
warnings.simplefilter(action='ignore')
```


```python
import subprocess
import sys

# If a package is not available call this function with the name of the missing library
def install(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
```


```python
install(["arviz","pymc3"])
```
