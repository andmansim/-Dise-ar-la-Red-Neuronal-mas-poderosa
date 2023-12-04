import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 24})

#Inicializo el dataset y imprimo sus características
dataset = TUDataset(root='.', name='PROTEINS').shuffle()
print(f'Dataset: {dataset}')
print('-------------------')
print(f'Número de grafos: {len(dataset)}')
print(f'Número de nodos: {dataset[0].x.shape[0]}')
print(f'Número de características: {dataset.num_features}')
print(f'Número de clases: {dataset.num_classes}')

#Dibujamos un grafo para ver cómo es:
# Convertimos el conjunto de datos en un grafo de NetworkX y lo hacemos no dirigido
G = to_networkx(dataset[2], to_undirected=True)

# Diseño tridimensional utilizando el algoritmo de spring layout
pos = nx.spring_layout(G, dim=3, seed=0)

# Extraemos las posiciones de nodos y aristas del diseño
node_xyz = np.array([pos[v] for v in sorted(G)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

# Creamos la figura tridimensional
fig = plt.figure(figsize=(16,16))
ax = fig.add_subplot(111, projection="3d")

# Suprimimos etiquetas de los ejes
for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
    dim.set_ticks([])

# Graficar los nodos - la transparencia (alpha) está escalada automáticamente por "profundidad"
ax.scatter(*node_xyz.T, s=500, c="#0A047A")

# Graficar las aristas
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:gray")

# Ajustar el diseño de la figura
# fig.tight_layout()
plt.show()

'''La representación anterior se genera aleatoriamente. Para obtener la representación correcta, habría que usar Alphafold'''

#Entrenamos el modelo:
#Creamos los sets de entrenamiento, validación y testeo
train_dataset = dataset[:int(len(dataset) * 0.8)]
val_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
test_dataset = dataset[int(len(dataset) * 0.9):]

print(f'Número de grafos de entrenamiento: {len(train_dataset)}')
print(f'Número de grafos de validación: {len(val_dataset)}')
print(f'Número de grafos de testeo: {len(test_dataset)}')

#Creamos los mini-batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#los imprimimos

print('\nTrain loader:')
for i, subgraph in enumerate(train_loader):
    print(f' - Subgraph {i}: {subgraph}')

print('\nValidation loader:')
for i, subgraph in enumerate(val_loader):
    print(f' - Subgraph {i}: {subgraph}')

print('\nTest loader:')
for i, subgraph in enumerate(test_loader):
    print(f' - Subgraph {i}: {subgraph}')

'''Con los mini-lotes (batches), aceleramos en entrenamiento'''

#Vamos a hacer una implementación de GIN con concatenación de incrustación de gráficos
#Luego la comparamos con un GCN
