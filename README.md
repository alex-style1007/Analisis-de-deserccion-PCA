# Análisis de Deserción en la Escuela de Ingeniería
En este análisis, se examina el fenómeno de la deserción estudiantil en la Escuela de Ingeniería de una universidad no especificada, con el objetivo de comprender los factores que influyen en la permanencia de los estudiantes en sus programas académicos. La deserción estudiantil es un problema importante que enfrentan muchas instituciones educativas, ya que puede tener repercusiones tanto para los estudiantes como para las universidades en términos de rendimiento académico, recursos financieros y reputación institucional.

## **Objetivo**
El objetivo principal del análisis es identificar los factores que pueden estar relacionados con la deserción estudiantil en la Escuela de Ingeniería. Para lograr este objetivo, se utilizan técnicas de análisis de datos, incluido el Análisis de Componentes Principales (PCA), para explorar y visualizar patrones en los datos relacionados con la permanencia estudiantil.
 ## **Proceso**
 El análisis comienza con la carga de los datos de deserción estudiantil, que pueden incluir información demográfica, académica y socioeconómica de los estudiantes. Luego, se realiza una exploración inicial de los datos, que incluye estadísticas descriptivas y visualizaciones preliminares para comprender la distribución de los datos y identificar posibles tendencias.

A continuación, se lleva a cabo el preprocesamiento de datos, que puede incluir la eliminación de columnas con poca información, la conversión de variables categóricas a numéricas y la normalización de los datos. Posteriormente, se realiza el análisis de componentes principales (PCA) para reducir la dimensionalidad de los datos y explorar la estructura subyacente de los mismos.

Finalmente, se realizan análisis adicionales con PCA, como la visualización de datos transformados y la identificación de variables importantes que contribuyen a la deserción estudiantil. Estos análisis proporcionan información valiosa que puede ser utilizada por la universidad para implementar estrategias efectivas de retención estudiantil y mejorar la experiencia educativa de sus estudiantes de ingeniería.

## Análisis de datos con Python

### Importar bibliotecas
Se importan las bibliotecas necesarias para el análisis de datos, incluyendo pandas, matplotlib, numpy, scikit-learn y seaborn.
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
import seaborn as sns
from sklearn.preprocessing import StandardScaler
```
### Carga de Datos 
Se cargan los datos desde un archivo CSV, se realiza una descripción estadística de los datos, se visualiza una columna y se muestran las primeras filas del DataFrame.
```python
data = pd.read_csv('https://dominio/carpeta/permanencia_ingenierias.csv', sep=';', encoding='latin_1', decimal=",")
data.describe()
data['ALERTASMANUALES'].plot()
data.head()
```
### Preprocesamiento de datos 
Se cargan los datos desde un archivo CSV, se realiza una descripción estadística de los datos, se visualiza una columna y se muestran las primeras filas del DataFrame.
```python
data['SEXO'].value_counts()
data = data.drop(['ALERTASACADEMICAS', 'ALERTASMANUALES'], axis=1)
data['DESERSIÓN'] = data['DESERSIÓN'].replace({'NO': 0, 'SI': 1})
data = pd.get_dummies(data, columns=['ULTIMO ESTADO ACADÉMICO'], drop_first=True)
data['SEXO'] = data['SEXO'].replace({'M': 0, 'F': 1, 'N': 2})
dataNormalizada = StandardScaler().fit_transform(data)
data = pd.DataFrame(data=dataNormalizada, columns=data.columns)
```
### Análisis de componentes principales (PCA) 
Se calculan y visualizan los componentes principales, se muestra la varianza explicada por cada componente principal y se realizan transformaciones de datos utilizando PCA.
```python
pca = decomposition.PCA()
pca.fit(data.values)
Y = pca.transform(data.values)
plt.plot(range(1, len(data.columns) + 1), np.cumsum(pca.explained_variance_ratio_), 'x-b')
plt.xlabel('Número de componentes')
plt.ylabel('Porcentaje de la varianza explicada')
plt.grid()
Y.head()
```
![pca](https://github.com/alex-style1007/Images/blob/main/PCA.png)
### Visualización de los datos transformados por PCA 
Se visualizan los datos transformados por PCA y se muestran los componentes principales en un gráfico.
```python
pca = decomposition.PCA(n_components=2)
X = data.drop('DESERSIÓN', axis=1).values
pca.fit(X)
Y = pca.transform(X)
plt.plot(Y[data['DESERSIÓN'] == 0, 0], Y[data['DESERSIÓN'] == 0, 1], 'ob', label='No deserción')
plt.plot(Y[data['DESERSIÓN'] == 1, 0], Y[data['DESERSIÓN'] == 1, 1], 'xr', label='Sí deserción')
plt.figure(figsize=(12, 12))
comps2 = pca.components_.T
for co in data.drop('DESERSIÓN', axis=1).columns:
    K = 1
    plt.plot([0, comps2[i, 0] * K], [0, comps2[i, 1] * K], '-')
    plt.text(comps2[i, 0] * K, comps2[i, 1] * K, co)
plt.grid()
plt.axis('equal')
plt.xlim(0, 0.4)
plt.ylim(-0.2, 0.2)
Y
```
![pca](https://github.com/alex-style1007/Images/blob/main/PCAD.png)
### Análisis adicional con PCA 
Se instala y utiliza la biblioteca 'pca' para realizar un análisis adicional con PCA, incluyendo la representación gráfica de los resultados.
```python
!pip install pca
```

```python
from pca import pca
model = pca(normalize=False)
X = data.drop('DESERSIÓN', axis=1).values
labels = data.drop('DESERSIÓN', axis=1).columns
Y = data['DESERSIÓN'].values
results = model.fit_transform(X, col_labels=labels, row_labels=Y)
model.scatter()
model.biplot()
```
![pca](https://github.com/alex-style1007/Images/blob/main/PCAf.png)
