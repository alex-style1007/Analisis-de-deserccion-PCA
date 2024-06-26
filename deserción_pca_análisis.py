import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('https://dominio/carpeta/permanencia_ingenierias.csv', sep=';', encoding='latin_1',decimal=",")
data.describe()
data['ALERTASMANUALES'].plot()
data.head()

data['SEXO'].value_counts()

data=data.drop(['ALERTASACADEMICAS','ALERTASMANUALES'], axis=1) #Se eliminan por poseer poca información. O son 0 o son muy pocos valores.
data['DESERSIÓN']=data['DESERSIÓN'].replace({'NO':0, 'SI':1})
des=data['DESERSIÓN'].values
data = pd.get_dummies(data, columns=['ULTIMO ESTADO ACADÉMICO'], drop_first=True)
data['SEXO']=data['SEXO'].replace({'M':0, 'F':1,'N':2})
data['APOYOS ECONÓMICO EN MATRICULA']=data['APOYOS ECONÓMICO EN MATRICULA'].replace({'NO':0, 'SI':1})
data['APOYO INSTITUCIONAL']=data['APOYO INSTITUCIONAL'].replace({'NO':0, 'SI':1})


dataNormalizada=StandardScaler().fit_transform(data)
data=pd.DataFrame(data=dataNormalizada, columns=data.columns)
data.head()

data.var()

data.cov()

#Los vectores y componentes principales se obtienen de la matriz de covarianza
np.linalg.eig(data.cov())

pca = decomposition.PCA()
pca.fit(data.values)
Y=pca.transform(data.values)
cols=data.columns
Y=pd.DataFrame(Y, columns=['PC_'+str(i) for i in range(1,len(cols)+1)])

plt.plot(range(1,len(cols)+1), np.cumsum(pca.explained_variance_ratio_),'x-b')
plt.xlabel('Número de componentes')
plt.ylabel('Porcentaje de la varianza')
plt.grid()
Y.head()

for k,v in zip(data.columns, pca.components_[20]):
  print(k,v)

pca = decomposition.PCA(n_components=2)
X=data.drop('DESERSIÓN', axis=1).values
cols=data.drop('DESERSIÓN', axis=1).columns
pca.fit(X)
Y=pca.transform(X)
plt.plot(Y[des==0,0],Y[des==0,1],'ob' , label='No deserción')
plt.plot(Y[des==1,0],Y[des==1,1],'xr' , label='SI deserción')
comps2=pca.components_.T

plt.figure(figsize=(12,12))
i=0
for co in data.drop('DESERSIÓN', axis=1).columns:
  K=1
  plt.plot([0,comps2[i,0]*K], [0,comps2[i,1]*K], '-')
  plt.text(comps2[i,0]*K,comps2[i,1]*K, co)
  i+=1
plt.grid()
plt.axis('equal')
print(Y)
plt.xlim(0,0.4)
plt.ylim(-0.2, 0.2)

Y

#!pip install pca

from pca import pca
model = pca(normalize=False)
# Fit transform and include the column labels and row labels
X=data.drop('DESERSIÓN', axis=1).values
labels=data.drop('DESERSIÓN', axis=1).columns
Y=data['DESERSIÓN'].values
results = model.fit_transform(X, col_labels=labels, row_labels=Y)
model.scatter()
model.biplot()
