from monty.serialization import loadfn, MontyDecoder
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from jarvis.core.atoms import Atoms
import torch
from torch_geometric.data import Data
#from ase import Atoms
from ase import neighborlist
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from torch_geometric.data import Dataset

dat_3d = loadfn('jdft_3d-12-12-2022.json',cls=MontyDecoder)

jid, formula, ehull, optb88vdw_bandgap, atoms, url =[],[],[],[],[],[]

for idx,i in enumerate(dat_3d):
    if idx<8000:
      jid.append(i['jid'])
      formula.append(i['formula'])
      ehull.append(float(i['ehull']))
      optb88vdw_bandgap.append(float(i['optb88vdw_bandgap']))
      atoms.append(i['atoms'])
      url.append(str("https://www.ctcms.nist.gov/~knc6/jsmol/")+str(i['jid'])+str(".html"))
    
df = pd.DataFrame([jid, formula, ehull, optb88vdw_bandgap, atoms, url])
df = df.transpose()
headers = ['jid', 'formula', 'ehull', 'optb88vdw_bandgap', 'atoms', 'url']
df.columns = headers


cols = ['jid','formula','atoms','url']

def clean_data(df,cols):
    df_dropped = df.drop(columns=cols).astype('float16')
    scaler = StandardScaler()
    standard = scaler.fit_transform(df_dropped)
    df_std = pd.DataFrame(standard, columns = df_dropped.columns, index= df_dropped.index )
    return df_std, scaler.mean_, scaler.scale_

df_clean, mean, scale = clean_data(df.copy(),cols)
df_non_numeric = df[cols]  
df_final = pd.concat([df_clean, df_non_numeric], axis=1)
df_final = df_final[df.columns]

for column in df_final[["ehull","optb88vdw_bandgap"]]:
    col_kurtosis = kurtosis(np.log1p(df_final[column]))
    col_skew = skew(np.log1p(df_final[column]))
    print(f"{column} - Skewness: {col_skew}, Kurtosis: {col_kurtosis}")

    plt.figure(figsize=[8,4])
    sns.violinplot(x=np.log1p(df_final[column]))
    plt.title(f'Violin plot of {column}')
    plt.xlabel(column)
    plt.savefig(f"Distribution_violin_{column}.png")


a = Atoms.from_dict(df_final["atoms"][0])
neigh = a.get_neighbors_cutoffs()

data_list = []
fails = 0

for kk, row in df_final.iterrows():  
    a = Atoms.from_dict(row["atoms"])  
    atoms = a.ase_converter(pbc=True)
    nl = neighborlist.NeighborList(neighborlist.natural_cutoffs(atoms))
    nl.update(atoms)

    atomic_numbers = atoms.get_atomic_numbers()

    try:
        temp_x = torch.tensor(atomic_numbers, dtype=torch.float).unsqueeze(1)
        temp_y = torch.tensor(np.log1p(row["ehull"]), dtype=torch.float)
        temp_pos = torch.tensor(atoms.get_positions(), dtype=torch.float)

        edge_index_list = []
        for i in range(len(atoms)):
            neighbors, offsets = nl.get_neighbors(i)
            for neighbor in neighbors:
                if neighbor != i:
                    edge_index_list.append([i, neighbor])

        # Ensure edge_index_list is not empty
        if len(edge_index_list) > 0:
            temp_edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        else:
            #temp_edge_index = torch.empty((2, 0), dtype=torch.long)
            fails += 1
            print(f"Skipping row {kk} due to no edges.")
            continue  # Skip this data point entirely

        # Ensure valid edge_index before calculating distances
        if temp_edge_index.shape[1] > 0:
            edge_vectors = temp_pos[temp_edge_index[0]] - temp_pos[temp_edge_index[1]]
            temp_edge_attr = torch.norm(edge_vectors, dim=1).unsqueeze(1)
        else:
            #temp_edge_attr = torch.empty((0, 1), dtype=torch.float)
            #fails += 1
            #print(f"Skipping row {kk} due to no edges.")
            continue  # Skip this data point entirely

    except Exception as e:
        fails += 1  # Count the failed samples
        print(f"Skipping row {kk} due to error: {e}")  # Debugging

    # edge_attr - fancier 
    data_list.append(Data(x=temp_x, pos=temp_pos, edge_index=temp_edge_index, edge_attr=temp_edge_attr, y=temp_y))

print(f"Done, failed entries: {fails}")

data_point = data_list[0]
print(f'Number of nodes: {data_point.num_nodes}')
print(f'Number of edges: {data_point.num_edges}')
print(f'Average node degree: {data_point.num_edges / data_point.num_nodes:.2f}')
print(f'Has isolated nodes: {data_point.has_isolated_nodes()}')
print(f'Has self-loops: {data_point.has_self_loops()}')
print(f'Is undirected: {data_point.is_undirected()}')

class JarvisDataset(Dataset):
    def __init__(self,data_list,transform=None,pre_transform=None):
        super().__init__(transform,pre_transform)
        self.data_list = data_list 

    def len(self):
        return len(self.data_list)
    def get(self,idx):
        return self.data_list[idx]
    def mean(self):
        return np.mean(np.array([data.y for data in data_list]))
    def scale(self):
        return np.std(np.array([data.y for data in data_list]))
    
dataset = JarvisDataset(data_list)