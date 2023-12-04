import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

