import torch 
from torch_geometric.data import Data, InMemoryDataset
from ase.io import read 
from ase.neighborlist import neighbor_list
import numpy as np
from config_loader import load_config

# MAKES EACH FRAME INTO A GRAPH

config = load_config()

PROCESSED_XYZ = config['data']['processed_xyz']
PROCESSED_PT = config['data']['processed_pt']
CUTOFF_RADIUS = config['graph']['cutoff_radius']
MAX_NEIGHBORS = config['graph']['max_neighbors']

# InMemoryDataset: PyTorch Geometric (PyG) dataset class for when the whole dataset can be stored in RAM efficiently
#   Required functions:
#       raw_file_names() --> returns raw file name with atomic information
#       processed_file_names(), --> returns processed file name as a .pt (Pytorch format for saved objects)
#       process() --> reads raw data, builds objects, collates, saves to processed paths
class GCGraphDataset(InMemoryDataset):
    def __init__(self, root, xyz_file=PROCESSED_XYZ, cutoff=CUTOFF_RADIUS, max_neighbors=MAX_NEIGHBORS, transform=None):
        self.raw_filename = xyz_file
        self.cutoff = cutoff # Neighbor cutoff distance used to build edges
        self.max_neighbors = max_neighbors # Paper explicitly restricts to 12 nearest neighbors
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False) # Loads cached processed dataset if available

    @property
    def raw_file_names(self):
        return [self.raw_filename]
    
    @property
    def processed_file_names(self):
        return [PROCESSED_PT]

    def process(self):
        print(f"Creating graph structure from {self.raw_filename}")
        frames = read(self.raw_filename, index=':')
        
        graph_structures = []

        for i, atoms in enumerate(frames):
            # Creates edge list 
            i_indices, j_indices, d_ij = neighbor_list('ijd', atoms, self.cutoff) # shape == [num_edges,1]
            
            new_i, new_j = [], []
            
            # Loop through every single atom in the current frame
            for node_idx in range(len(atoms)):
                # Create a boolean mask to find all neighbors connected to node_idx
                mask = (i_indices == node_idx)
                node_j = j_indices[mask]
                node_d = d_ij[mask]
                
                # If the atom has more than 12 neighbors within the 5.0A cutoff, sort them by distance (closest first) and keep only the top 12.
                if len(node_d) > self.max_neighbors:
                    # np.argsort returns the indices that would sort the array
                    sorted_indices = np.argsort(node_d)[:self.max_neighbors]
                    node_j = node_j[sorted_indices]
                    
                new_i.extend([node_idx] * len(node_j))
                new_j.extend(node_j)
                
            # Create the final edge_index tensor with the filtered neighbors
            edge_index = torch.tensor([new_i, new_j], dtype=torch.long) # edge_index.shape == [2,num_filtered_edges (12)]
            
            # Validate that this is a carbon-only structure
            atomic_numbers = atoms.get_atomic_numbers()
            if not np.all(atomic_numbers == 6):
                 raise ValueError(
                      f"Frame {i} contains non-carbon atoms: "
                      f"{sorted(set(atomic_numbers))}"
                      )

            # Extracts the forces, energy, and stress from the ASE Atoms object
            z = torch.zeros(len(atoms), dtype=torch.long)
            forces = torch.tensor(
                atoms.get_forces(),
                dtype=torch.float,
                )
            energy = torch.tensor(
                [atoms.get_potential_energy()],
                dtype=torch.float,
                )
            pos = torch.tensor(
                 atoms.get_positions(),
                 dtype=torch.float,
                )
            density = torch.tensor(
                [atoms.info["density"]],
                dtype=torch.float,
                )
            
            # Extracts the 3D cell dimensions (Lengths of the simulation box) --> diffusion model will need this to wrap noisy atoms back into the box via PBC
            cell_dims = atoms.cell.lengths()
            cell = torch.tensor(cell_dims, dtype=torch.float).unsqueeze(0) # Shape [1, 3]

            graph = Data(
                pos=pos, 
                z=z, 
                edge_index=edge_index, 
                forces=forces, 
                energy=energy,
                y=density,
                cell=cell
            )
            graph_structures.append(graph)

        graphs, slices = self.collate(graph_structures) # 1 stacked Data object containing all graphs concatenated and a dictionary of slices
        torch.save((graphs, slices), self.processed_paths[0])
        print(f"Success! Processed {len(graph_structures)} graphs.")


if __name__ == "__main__":
    graph_dataset = GCGraphDataset(root='.', xyz_file=PROCESSED_XYZ)
    sample = graph_dataset[0]
    
    print(sample)
    print(f"Positions shape: {sample.pos.shape}")
    print(f"Density: {sample.y.item():.4f} g/cm³")
    print(f"Cell dimensions: {sample.cell}")