import numpy as np
import matplotlib as pl
import pandas as pd
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None,sep=',');
df.columns = ['sepal_len','sepal_wid','petal_len','petal_wid','class']
df.dropna(how="all",inplace=True)
df.tail()
X = df.iloc[:,0:4].values
print(X)
Y = df.iloc[:,4].values
print(Y)
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
X_cov = np.transpose(X).dot(X)
print(X_cov)
eig_vals, eig_vecs = np.linalg.eig(X_cov)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
sum_of_ev = 0
for i in eig_vals:
    sum_of_ev = sum_of_ev+i
var1 = eig_vals[0]/sum_of_ev
var2 = eig_vals[1]/sum_of_ev
var3 = eig_vals[2]/sum_of_ev
var4 = eig_vals[3]/sum_of_ev
print ('Due to PC1 : %s ' %(var1*100))
print ('Due to PC2 : %s ' %(var2*100))
print ('Due to PC3 : %s ' %(var3*100))
print ('Due to PC4 : %s ' %(var4*100))
W = np.transpose([eig_vecs[:,0],eig_vecs[:,1]])
print (np.matrix(W))
T = X.dot(W)
print (np.matrix(T))
