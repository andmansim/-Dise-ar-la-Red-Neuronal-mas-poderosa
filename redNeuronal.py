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

class GCN(torch.nn.Module):
    """GCN (Red de Convolución de Grafos)"""
    def __init__(self, dim_h):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(dim_h, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Obtener incrustaciones de nodos
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)

        # Resumen del grafo a nivel global
        hG = global_mean_pool(h, batch)

        # Clasificador
        h = F.dropout(hG, p=0.5, training=self.training)
        h = self.lin(h)
        
        return hG, F.log_softmax(h, dim=1)

class GIN(torch.nn.Module):
    """GIN (Red de Neuronas Isomórficas)"""
    def __init__(self, dim_h):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(dataset.num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Obtener incrustaciones de nodos
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Resumen del grafo a nivel global
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenar incrustaciones de grafos
        h = torch.cat((h1, h2, h3), dim=1)

        # Clasificador
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h, F.log_softmax(h, dim=1)

gcn = GCN(dim_h=32)
gin = GIN(dim_h=32)


def train(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                      lr=0.01,
                                      weight_decay=0.01)
    epochs = 100    
    model.train()
    for epoch in range(epochs+1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        # Entrenar en lotes
        for data in loader:
          optimizer.zero_grad()
          _, out = model(data.x, data.edge_index, data.batch)
          loss = criterion(out, data.y)
          total_loss += loss / len(loader)
          acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
          loss.backward()
          optimizer.step()

          # Validación
          val_loss, val_acc = test(model, val_loader)

    # Imprimir métricas cada 10 epochs
    if(epoch % 10 == 0):
        print(f'Época {epoch:>3} | Pérdida Entrenamiento: {total_loss:.2f} '
              f'| Precisión Entrenamiento: {acc*100:>5.2f}% '
              f'| Pérdida Validación: {val_loss:.2f} '
              f'| Precisión Validación: {val_acc*100:.2f}%')
              
    test_loss, test_acc = test(model, test_loader)
    print(f'Pérdida Prueba: {test_loss:.2f} | Precisión Prueba: {test_acc*100:.2f}%')
    
    return model

@torch.no_grad()
def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0
  
    for data in loader:
        _, out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
   
    return loss, acc

def accuracy(pred_y, y):
    """Calcular precisión."""
    return ((pred_y == y).sum() / len(y)).item()

gcn = train(gcn, train_loader)
gin = train(gin, train_loader)

#visualizamos las proteínas que hemos clasificado con el GIN y el GCN:
#GCN
fig, ax = plt.subplots(4, 4, figsize=(16,16))
fig.suptitle('GCN - Graph classification')
for i, data in enumerate(dataset[1113-16:]):    
    # Calculamos el color (verde si está correcto, rojo cualquier otra cosa)
    _, out = gcn(data.x, data.edge_index, data.batch)    
    color = "green" if out.argmax(dim=1) == data.y else "red"
    # Gráfico:
    ix = np.unravel_index(i, ax.shape)    
    ax[ix].axis('off')    
    G = to_networkx(dataset[i], to_undirected=True)    
    nx.draw_networkx(G,                    
                     pos=nx.spring_layout(G, seed=0),
                     with_labels=False,                    
                     node_size=150,                    
                     node_color=color,                    
                     width=0.8,                    
                     ax=ax[ix]                   
                     )

fig, ax = plt.subplots(4, 4, figsize=(16,16))
fig.suptitle('GIN - Graph classification')

#GIN
for i, data in enumerate(dataset[1113-16:]):
    # Calculamos el color (verde si está correcto, rojo cualquier otra cosa)
    _, out = gin(data.x, data.edge_index, data.batch)
    color = "green" if out.argmax(dim=1) == data.y else "red"

    #Gráfico    
    ix = np.unravel_index(i, ax.shape)
    ax[ix].axis('off')
    G = to_networkx(dataset[i], to_undirected=True)
    nx.draw_networkx(G,
                    pos=nx.spring_layout(G, seed=0),
                    with_labels=False,
                    node_size=150,
                    node_color=color,
                    width=0.8,
                    ax=ax[ix]
                    )

#Combinamos las incrustaciones de gráficos, tomando la media de los vectores de salida normalizados:
gcn.eval()
gin.eval()
acc_gcn = 0
acc_gin = 0
acc = 0

for data in test_loader:
    # Cogemos las clasificaciones de cada modelo
    _, out_gcn = gcn(data.x, data.edge_index, data.batch)
    _, out_gin = gin(data.x, data.edge_index, data.batch)
    out = (out_gcn + out_gin)/2

    # Calculamos los valores de precisión
    acc_gcn += accuracy(out_gcn.argmax(dim=1), data.y) / len(test_loader)
    acc_gin += accuracy(out_gin.argmax(dim=1), data.y) / len(test_loader)
    acc += accuracy(out.argmax(dim=1), data.y) / len(test_loader)

# Imprimimos los resultados
print(f'GCN accuracy:     {acc_gcn*100:.2f}%')
print(f'GIN accuracy:     {acc_gin*100:.2f}%')
print(f'GCN+GIN accuracy: {acc*100:.2f}%')
