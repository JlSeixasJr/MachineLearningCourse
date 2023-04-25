# NonParametric Methods

## Lazy learner

* The method does not keep calculated values (parameters).
* Does not "learn".


```python
import sklearn as skl
import matplotlib as mpl

print("Scikit Learn version: "+ skl.__version__)
print("Matplotlib version: "+ mpl.__version__)
```

    Scikit Learn version: 1.0.2
    Matplotlib version: 3.5.1



```python
import matplotlib.pyplot as plt

from sklearn import datasets

x, y = datasets.make_blobs(n_samples=600, n_features=2, cluster_std=1, random_state=1)

plt.scatter(x[:,0], x[:,1], c='k', marker='+', s=50)
plt.show()
```


    
![png](output_3_0.png)
    



```python
x, y = datasets.make_blobs(n_samples=600, n_features=2, cluster_std=3, random_state=1)

plt.scatter(x[:,0], x[:,1], c='k', marker='+', s=50)
plt.show()
```


    
![png](output_4_0.png)
    


## K Nearest Neighbors


```python
import mlxtend as mlx

print("ML Extend version: "+ mlx.__version__)
```

    ML Extend version: 0.19.0



```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(algorithm='ball_tree') #n_neighbors = 5 algorithm='auto'

knn.fit(x, y)

from mlxtend.plotting import plot_decision_regions

fig, ax = plt.subplots()

plot_decision_regions(x, y, clf=knn, legend=2, ax=ax)
```




    <AxesSubplot:>




    
![png](output_7_1.png)
    



```python
knn = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')

knn.fit(x, y)

fig, ax = plt.subplots()

plot_decision_regions(x, y, clf=knn, legend=2, ax=ax)
```




    <AxesSubplot:>




    
![png](output_8_1.png)
    



```python
knn = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')

knn.fit(x, y)

fig, ax = plt.subplots()

plot_decision_regions(x, y, clf=knn, legend=2, ax=ax)
```




    <AxesSubplot:>




    
![png](output_9_1.png)
    



```python
knn = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')

knn.fit(x, y)

fig, ax = plt.subplots()

plot_decision_regions(x, y, clf=knn, legend=2, ax=ax)
```




    <AxesSubplot:>




    
![png](output_10_1.png)
    



```python
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, stratify=y)

knn = KNeighborsClassifier()

knn.fit(xTrain, yTrain)

fig, ax = plt.subplots()

plot_decision_regions(xTrain, yTrain, clf=knn, legend=2, ax=ax)
plt.scatter(xTest[:,0], xTest[:,1], c='k', marker='+', s=50)
```




    <matplotlib.collections.PathCollection at 0x7fb09a4ff350>




    
![png](output_11_1.png)
    



```python
from sklearn import metrics
from sklearn import model_selection

knnPred = knn.predict(xTest) #Shape => (200,1) [2]
knnProb = knn.predict_proba(xTest) #Shape => (200, 3) [0.1, 0.4, 0.5]

print("Accuracy: ", metrics.balanced_accuracy_score(yTest, knnPred))
print("F1: ", metrics.f1_score(yTest, knnPred, average='weighted'))
print("Loss: ", metrics.log_loss(yTest, knnProb))
print("Cross validation: ", model_selection.cross_val_score(knn, x, y))

print(metrics.classification_report(yTest, knnPred))
```

    Accuracy:  0.8333333333333334
    F1:  0.8355347959293216
    Loss:  1.7776799667438365
    Cross validation:  [0.88333333 0.78333333 0.875      0.85833333 0.825     ]
                  precision    recall  f1-score   support
    
               0       1.00      0.93      0.97        60
               1       0.75      0.78      0.76        60
               2       0.77      0.78      0.78        60
    
        accuracy                           0.83       180
       macro avg       0.84      0.83      0.84       180
    weighted avg       0.84      0.83      0.84       180
    



```python
metrics.plot_confusion_matrix(knn, xTest, yTest)
```

    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)





    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7fb093fcd210>




    
![png](output_13_2.png)
    


<img src="BiasVariance.png">

## Prototype kMeans


```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(xTrain)

plt.scatter(xTrain[y_km==0, 0], xTrain[y_km==0, 1], s=40, c='r', marker='s', label='Cluster 1')
plt.scatter(xTrain[y_km==1, 0], xTrain[y_km==1, 1], s=40, c='g', marker='o', label='Cluster 2')
plt.scatter(xTrain[y_km==2, 0], xTrain[y_km==2, 1], s=40, c='b', marker='v', label='Cluster 3')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=200, c='k', marker='X', label='Centroids')

plt.legend(scatterpoints=1)
```




    <matplotlib.legend.Legend at 0x7fb0993fff50>




    
![png](output_16_1.png)
    



```python
centroids = knn.predict(km.cluster_centers_)

print(km.cluster_centers_, centroids)
```

    [[ -1.44257123   4.49625692]
     [ -6.69989268  -8.05645411]
     [-10.82197375  -2.89645942]] [0 2 1]



```python
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(km.cluster_centers_, centroids)

fig, ax = plt.subplots()

plot_decision_regions(km.cluster_centers_, centroids, clf=knn, legend=2, ax=ax)
plt.scatter(xTest[:,0], xTest[:,1], c='k', marker='+', s=50)
```




    <matplotlib.collections.PathCollection at 0x7fb099519ed0>




    
![png](output_18_1.png)
    



```python
knnPred = knn.predict(xTest) #Shape => (200,1) [2]
knnProb = knn.predict_proba(xTest) #Shape => (200, 3) [0.1, 0.4, 0.5]

print("Accuracy: ", metrics.balanced_accuracy_score(yTest, knnPred))
print("F1: ", metrics.f1_score(yTest, knnPred, average='weighted'))
print("Loss: ", metrics.log_loss(yTest, knnProb))
print("Cross validation: ", model_selection.cross_val_score(knn, x, y))

print(metrics.classification_report(yTest, knnPred))
```

    Accuracy:  0.8444444444444444
    F1:  0.844455064068502
    Loss:  5.37269855031944
    Cross validation:  [0.8        0.76666667 0.8        0.78333333 0.84166667]
                  precision    recall  f1-score   support
    
               0       0.98      0.95      0.97        60
               1       0.81      0.72      0.76        60
               2       0.75      0.87      0.81        60
    
        accuracy                           0.84       180
       macro avg       0.85      0.84      0.84       180
    weighted avg       0.85      0.84      0.84       180
    



```python
metrics.plot_confusion_matrix(knn, xTest, yTest)
```

    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)





    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7fb099526e90>




    
![png](output_20_2.png)
    



```python
km = KMeans(n_clusters=12, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(xTrain)

plt.scatter(xTrain[y_km==0, 0], xTrain[y_km==0, 1], s=40, c='r', marker='s', label='Cluster 1')
plt.scatter(xTrain[y_km==1, 0], xTrain[y_km==1, 1], s=40, c='g', marker='o', label='Cluster 2')
plt.scatter(xTrain[y_km==2, 0], xTrain[y_km==2, 1], s=40, c='b', marker='v', label='Cluster 3')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=200, c='k', marker='X', label='Centroids')

plt.legend(scatterpoints=1)
```




    <matplotlib.legend.Legend at 0x7fb099f9b790>




    
![png](output_21_1.png)
    



```python
centroids = knn.predict(km.cluster_centers_)

print(km.cluster_centers_, centroids)
```

    [[-10.37847783  -0.03464147]
     [ -2.7103044   -6.76868196]
     [  1.14459309   6.44878167]
     [ -7.20375609  -4.38163368]
     [ -4.57946963   7.12256613]
     [-10.07129797  -8.86252599]
     [ -6.75535616  -8.29301884]
     [-14.97470068  -5.68816111]
     [-11.49700134  -3.73165401]
     [  0.55912526   1.226112  ]
     [ -5.67474522 -11.75030839]
     [ -2.52632533   3.3192075 ]] [1 2 0 2 0 2 2 1 1 0 2 0]



```python
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(km.cluster_centers_, centroids)

fig, ax = plt.subplots()

plot_decision_regions(km.cluster_centers_, centroids, clf=knn, legend=2, ax=ax)
plt.scatter(xTest[:,0], xTest[:,1], c='k', marker='+', s=50)
```




    <matplotlib.collections.PathCollection at 0x7fb0996b7650>




    
![png](output_23_1.png)
    



```python
knnPred = knn.predict(xTest) 
knnProb = knn.predict_proba(xTest)

print("Accuracy: ", metrics.balanced_accuracy_score(yTest, knnPred))
print("F1: ", metrics.f1_score(yTest, knnPred, average='weighted'))
print("Loss: ", metrics.log_loss(yTest, knnProb))
print("Cross validation: ", model_selection.cross_val_score(knn, x, y))

print(metrics.classification_report(yTest, knnPred))
```

    Accuracy:  0.811111111111111
    F1:  0.8088132128357253
    Loss:  6.523991096816462
    Cross validation:  [0.8        0.76666667 0.8        0.78333333 0.84166667]
                  precision    recall  f1-score   support
    
               0       1.00      0.93      0.97        60
               1       0.83      0.58      0.69        60
               2       0.67      0.92      0.77        60
    
        accuracy                           0.81       180
       macro avg       0.83      0.81      0.81       180
    weighted avg       0.83      0.81      0.81       180
    



```python
metrics.plot_confusion_matrix(knn, xTest, yTest)
```

    /Users/joseluis/opt/anaconda3/lib/python3.7/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)





    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7fb09a193610>




    
![png](output_25_2.png)
    

