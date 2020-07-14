
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
#-----------------------------------------------------------------------------
#Observo correlacion (YPF ; WTI)
data = pd.read_excel("Oil.xlsx")
papel = "YPF"
Px = np.array(list(zip(data['WTI Px'], data[papel + " Px"])))
Var = np.array(list(zip(data['WTI Var'], data[papel + " Var"])))
print("Correlación",papel,"R^2 =",data['WTI Var'].corr(data[papel+' Var']).round(4))
#-----------------------------------------------------------------------------
# Busco k clusters con métodos
#K-Means
inertias = []
K = range(2, 10)

for k in K:
    model = KMeans(n_clusters=k)
    model.fit(Px)
    inertias.append(model.inertia_)

plt.plot(K, inertias, '-o')
plt.title('Elbow Method For k*')
plt.xlabel('N° of clusters, k')
plt.ylabel('Inertia')
plt.xticks(K)
plt.show()

#Silhouette Method
#El coeficiente de silueta para una muestra es (b - a) / max(a, b)
#"a" es la distacia media intra-cluster
#"b" es la distacia media del cluster mas cercano
sil = []

for k in K:
  model = KMeans(n_clusters = k)
  model.fit(Px)
  labels = model.labels_
  sil.append(silhouette_score(Px, labels, metric = 'euclidean'))

plt.plot(K, sil, '-o')
plt.title('Silhouette Method For k*')
plt.xlabel('N° of clusters, k')
plt.ylabel('Silhouette Score')
plt.xticks(K)
plt.show()
#-----------------------------------------------------------------------------
# Parametrización
papel = "YPF" # Disponibles: XOM RDSa BP PAM YPF CVX COP EOG SLB VLO WMB
n = 4 # k Clusters verificado con Elbow & Silhouette Methods
mCorr= "pearson" # Métodos (Pearson, Spearman, Kendall) 

# Detalles del Gráfico
plt.style.use('dark_background')
plt.figure(figsize=(12,6))
colores= ["orange","green","brown","cyan","magenta","blue","red","yellow","lightgreen","pink"]  
plt.title('Clusterización (n=' +str(n)+ ') de correlacíón (' + mCorr + ') WTI-'+papel, fontsize=13)
plt.xlabel('WTI USD', fontsize=10)
plt.ylabel(papel+' USD', fontsize=10)

# Clustering 2D points
model = KMeans(n_clusters=n)
labels = model.fit_predict(Px)

kList = list(labels)

for i in range(len(data)):
    data.loc[i,'cluster'] = kList.pop()

for c in range(n):
    df = data.loc[data.cluster == c]
    co = round(df['WTI Var'].corr(df[papel + " Var"]),2)
    plt.scatter(Px[labels==c,0],Px[labels==c,1], s=1, color=colores[c])
    coords = (model.cluster_centers_[c,0] - data['WTI Px'].mean()*0.1, model.cluster_centers_[c,1])
    plt.gca().annotate("R^2 ="+str(co), coords, fontsize=13, c="white")

plt.show()
#-----------------------------------------------------------------------------
#Clustering Stocks
tickers = ['WTI', 'XOM', 'RDSa', 'BP', 'PAM', 'YPF', 'CVX', 'COP',
           'EOG', 'SLB', 'VLO', 'WMB']

companies = []

for i in range(len(tickers)):
    ticker = tickers[i] + ' Px'
    companies.append(ticker)

#Importación  
data.set_index('Date', inplace=True)
data = data[companies]   
dataT = data.T

#Arrays
array = np.array(dataT)

#Normalizacion
normalizer = Normalizer()
array_norm =  normalizer.transform(array)

# KMeans
inertias_ = []
K_ = range(2, 10)

for k in K_:
    model = KMeans(n_clusters=k)
    model.fit(array_norm)
    inertias_.append(model.inertia_)

plt.plot(K_, inertias_, '-o')
plt.title('Elbow Method For k*')
plt.xlabel('N° of clusters, k')
plt.ylabel('Inertia')
plt.xticks(K_)
plt.show()

#Silhouette Method
sil_ = []

for k in K_:
  model = KMeans(n_clusters = k)
  model.fit(array_norm)
  labels_ = model.labels_
  sil_.append(silhouette_score(dataT, labels_, metric = 'euclidean'))

plt.plot(K_, sil_, '-o')
plt.title('Silhouette Method For k*')
plt.xlabel('N° of clusters, k')
plt.ylabel('Silhouette Score')
plt.xticks(K_)
plt.show()

#-----------------------------------------------------------------------------
#Pipeline
kmeans = KMeans(n_clusters=7)
pipeline = make_pipeline(normalizer,kmeans)
pipeline.fit(array)   
    
#Busco los Clusters y visualizo que Stocks se mueven juntas
label_pip = pipeline.predict(array)
df = pd.DataFrame({'labels': label_pip, 'companies': companies})
print(df.sort_values('labels')) 
    
