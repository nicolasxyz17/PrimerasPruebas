#!/usr/bin/env python3
"""
"""
#-----------------------------------------------------------------------------
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

papel = "YPF"
data = pd.read_excel("Oil.xlsx")
Px = np.array(list(zip(data['WTI Px'], data[papel + " Px"])))
Var = np.array(list(zip(data['WTI Var'], data[papel + " Var"])))
print("Correlación",papel,"R^2 =",data['WTI Var'].corr(data[papel+' Var']).round(4))

# Clustering 2D points
model = KMeans(n_clusters=3)
model.fit(Px)
labels = model.predict(Px)

#How Many Clusters?
xs = Px[:,0]
ys = Px[:,1]
plt.scatter(xs,ys,c=labels, alpha=0.2)

centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

plt.scatter(centroids_x,centroids_y, marker='D', s=30)
plt.show()

# K vs inertia
ks = range(1, 15)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(Px)
    inertias.append(model.inertia_)

plt.plot(ks, inertias, '-o')
plt.xlabel('N° of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
#-----------------------------------------------------------------------------
# Parametrización
papel = "YPF" # Disponibles: XOM RDSa BP PAM YPF CVX COP EOG SLB VLO WMB
n = 5 # Cantidad de Clusters: Cualquier valor entero de 2 a 10
mCorr= "pearson" # pearson, spearman, kendall (son métodos de regresión)

# Detalles del Gráfico
plt.style.use('dark_background')
plt.figure(figsize=(12,6))
colores= ["orange","green","brown","cyan","magenta","blue","red","yellow","lightgreen","pink"]  
plt.title('Clusterización (n=' +str(n)+ ') de correlacíón (' + mCorr + ') WTI-'+papel, fontsize=13)
plt.xlabel('WTI USD', fontsize=10)
plt.ylabel(papel+' USD', fontsize=10)

# Clustering 2D points
km = KMeans(n_clusters=n, n_init=15)
lab = km.fit_predict(Px)

kList = list(lab)

for i in range(len(data)):
    data.loc[i,'cluster'] = kList.pop()

for c in range(n):
    df = data.loc[data.cluster == c]
    co = round(df['WTI Var'].corr(df[papel + " Var"]),2)
    plt.scatter(Px[lab==c,0],Px[lab==c,1], s=1, color=colores[c])
    coords = (km.cluster_centers_[c,0] - data['WTI Px'].mean()*0.1, km.cluster_centers_[c,1])
    plt.gca().annotate("R^2 ="+str(co), coords, fontsize=13, c="y")

plt.show()
#-----------------------------------------------------------------------------
#Clustering Stocks
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

tickers = ['WTI', 'XOM', 'RDSa', 'BP', 'PAM', 'YPF', 'CVX', 'COP',
           'EOG', 'SLB', 'VLO', 'WMB']

companies = []

for i in range(len(tickers)):
    ticker = tickers[i] + ' Px'
    companies.append(ticker)

#Importación  
data = pd.read_excel("Oil.xlsx")
data.set_index('Date', inplace=True)
data = data[companies]   
dataT = data.T

#Arrays
array = np.array(dataT)

#Pipeline
normalizer = Normalizer()
kmeans = KMeans(n_clusters=10)
pipeline = make_pipeline(normalizer,kmeans)
pipeline.fit(array)   
    
#Busco los Clusters y visualizo que Stocks se mueven juntas
labels = pipeline.predict(array)
df = pd.DataFrame({'labels': labels, 'companies': companies})
print(df.sort_values('labels')) 
    




 