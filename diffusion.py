import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import Batch
from tqdm import tqdm
from config_loader import load_config

config = load_config()

CUTOFF_RADIUS = config['graph']['cutoff_radius']
MAX_NEIGHBORS = config['graph']['max_neighbors']
NUM_STEPS = config['diffusion']['num_steps']
SIGMA_MIN = config['diffusion']['sigma_min']
SIGMA_MAX = config['diffusion']['sigma_max']
LANGEVIN_STEPS = config['diffusion']['langevin_steps']

# Builds graph edges dynamically using PBC and enforces the 5.0A cutoff and 12-neighbor limit
def build_pbc_graph(pos, cell, cutoff=CUTOFF_RADIUS, max_neighbors=MAX_NEIGHBORS):
    # pos shape: [N, 3] where N is the number of atoms
    # cell shape: [3] representing orthogonal box dimensions (Lx, Ly, Lz)
    
    # Gets every atom's 3D distance to every other atom
    dist_vec = pos.unsqueeze(1) - pos.unsqueeze(0) # [N, 1, 3] - [1, N, 3] = [N, N, 3]
    
    # Wraps distances so atoms interact across the box boundary if it's the shortest path
    dist_vec = dist_vec - torch.round(dist_vec / cell) * cell
    dist_sq = (dist_vec ** 2).sum(dim=-1) # [N, N] --> Square the distances for faster compute (avoids expensive sqrt)
    
    # Ignores self-interaction by setting diagonal to infinity so it is never picked as a neighbor
    dist_sq.fill_diagonal_(float('inf')) 
    
    num_atoms = pos.shape[0]
    
    # Vectorized Top-K: Finds the closest max_neighbors for ALL atoms simultaneously
    k = min(max_neighbors, num_atoms - 1)
    
    # topk with largest=False gets the SMALLEST distances for the entire batch at once
    dists, indices = torch.topk(dist_sq, k, dim=1, largest=False) # dists shape: [N, k], indices shape: [N, k]
    
    # Creates source and destination index tensors
    src = torch.arange(num_atoms, device=pos.device).view(-1, 1).expand(-1, k) # src shape: [N, k]
    dst = indices # dst shape: [N, k]
    
    # Flattens the tensors to 1D arrays to prepare for the cutoff mask
    src = src.flatten() # shape: [N * k]
    dst = dst.flatten() # shape: [N * k]
    dists = dists.flatten() # shape: [N * k]
    
    # Applies the 5.0 Å cutoff mask to remove neighbors that are too far away
    mask = dists < (cutoff ** 2)
    src = src[mask]
    dst = dst[mask]
    
    # Handle edge case where atoms are too far apart to form any edges
    if src.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device) # [2, 0]
        
    # Combines into final edge_index tensor
    edge_index = torch.stack([src, dst], dim=0) # Final edge_index shape: [2, E_final]
    return edge_index


# Main DDPM implementation
class DiffusionModel(nn.Module):
    def __init__(self, model, num_steps=NUM_STEPS, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, device='cpu'):
        super().__init__()
        self.model = model.to(device)
        self.num_steps = num_steps
        self.device = device
        
        # Training noise schedule [0.001 to 0.75 Å] linearly spaced
        self.sigma_train = torch.linspace(sigma_min, sigma_max, num_steps).to(device) # self.sigma_train shape: [3000]
    
    # Forward Process: Adds physical displacement noise
    def q_sample(self, x_0, t): # x_0 (pos coordinates) shape: [N, 3], t shape: [N] (Timestep index)
        
        # Move the noise schedule to the same device as the timestep index before slicing
        sigma_t = self.sigma_train.to(t.device)[t][:, None] # sigma_t shape: [N, 1]   
        noise = torch.randn_like(x_0) * sigma_t # Generates Gaussian noise --> [N, 3]
        
        # Returns corrupted coordinates and the exact noise added (target for the loss function)
        return x_0 + noise, noise

    def get_loss(self, data):
        device = data.pos.device
        x_0 = data.pos # [N_total, 3] where N_total is the sum of all atoms across the batch
        batch_size = data.num_graphs
        
        # Sample one random discrete timestep per graph in the batch
        t = torch.randint(0, self.num_steps, (batch_size,), device=device).long() # t shape: [Batch]
        
        # Broadcast the graph-level timestep to every individual atom in that graph
        t_per_atom = t[data.batch] # t_per_atom shape: [N_total]
        
        # Corrupt positions
        x_t, actual_displacement = self.q_sample(x_0, t_per_atom) # x_t shape: [N_total, 3], actual_displacement shape: [N_total, 3]
        
        # Updates data object with noisy positions to preserve original data.pos structure
        noisy_data = data.clone()
        noisy_data.pos = x_t
        
        # Rebuild graph edges for new noisy positions
        edge_indices = []
        batch_offsets = 0 # Tracks node index offsets because graphs are concatenated in one big batch
        
        for i in range(batch_size):
            mask = data.batch == i
            pos_i = x_t[mask] # [N_graph, 3]
            
            # Extracts cell dimensions from the dataset 
            cell_i = data.cell[i].squeeze() if hasattr(data, 'cell') else torch.tensor([10.0, 10.0, 10.0], device=device) # cell_i shape: [3]
            edges_i = build_pbc_graph(pos_i, cell_i, cutoff=CUTOFF_RADIUS, max_neighbors=MAX_NEIGHBORS) # [2, E_graph]
            
            # Add the offset to node indices so they point to the correct atoms in the concatenated batch
            edge_indices.append(edges_i + batch_offsets)
            batch_offsets += pos_i.shape[0]
            
        # Combine all batch edges into one tensor
        noisy_data.edge_index = torch.cat(edge_indices, dim=1) # noisy_data.edge_index shape: [2, E_total]
        
        # Normalize timestep for NequIP time embedder (between 0.0 and 1.0) to know how much noise was added
        t_norm = t.float() / self.num_steps # t_norm shape: [Batch]
        
        # GNN Predicts the physical noise that was added
        predicted_displacement = self.model(noisy_data, t_norm) # [N_total, 3]
        return F.mse_loss(predicted_displacement, actual_displacement) # MSE loss between predicted displacement and actual added displacement

    # Generates amorphous solids from scratch
    @torch.no_grad() # Sampling is for inference, not training so remove gradient tracking
    def sample(self, num_atoms, cell_dims, atom_types, target_density): # cell_dims: List of [Lx, Ly, Lz] 

        cell_tensor = torch.tensor(cell_dims, device=self.device, dtype=torch.float32) # cell_tensor shape: [3]
        f_coords = torch.rand(num_atoms, 3, device=self.device) # f_coords shape: [N, 3] --> # Initialize with uniform fractional coordinates between 0.0 and 1.0
        x = f_coords * cell_tensor # x shape: [N, 3] --> Scale to absolute Cartesian coordinates filling the simulation box
        z = torch.tensor(atom_types, device=self.device, dtype=torch.long) # z shape: [N] (e.g., Atomic numbers like 14 for Si, 6 for C)
        
        # Langevin extra noise schedule for generation (Decays from 1.0 to 0.0 over 2900 steps)
        sigma_gen = torch.linspace(1.0, 0.0, LANGEVIN_STEPS).to(self.device) # sigma_gen shape: [2900]
        
        # Reverse process: Step backward from T=2999 down to T=0
        for i in tqdm(reversed(range(self.num_steps)), total=self.num_steps):
            
            x = x - torch.floor(x / cell_tensor) * cell_tensor # Forces every coordinate into [0,L] --> wraps back into cell boundaries
            edge_index = build_pbc_graph(x, cell_tensor, cutoff=CUTOFF_RADIUS, max_neighbors=MAX_NEIGHBORS) # Builds dynamic edges for current positions; edge_index shape: [2, E]
            data = Data(pos=x, z=z, edge_index=edge_index)
            data.batch = torch.zeros(num_atoms, dtype=torch.long, device=self.device) # Shape [N]
            
            # Conditions structure on target density
            data.y = torch.tensor([target_density], device=self.device, dtype=torch.float32) # data.y shape: [1]
            
            # Model predicts structural noise 
            t_norm = torch.tensor([i / self.num_steps], device=self.device, dtype=torch.float32) # t_norm shape: [1] --> tells the model if this is a high noise graph or low noise
            predicted_displacement = self.model(data, t_norm) # shape: [N, 3]
            x = x - predicted_displacement # Removes predicted noise
            
            # Dynamically calculate the extra noise threshold and indices based on config variables
            # Langevin dynamics: extra noise to help escape local minima
            cutoff_step = self.num_steps - LANGEVIN_STEPS
            if i >= cutoff_step:
                idx = (self.num_steps - 1) - i
                extra_noise = torch.randn_like(x) * sigma_gen[idx] # extra_noise shape: [N, 3]
                x = x + extra_noise
                
        # Final PBC wrap to guarantee bounds
        x = x - torch.floor(x / cell_tensor) * cell_tensor
        return x
    
    # PyTorch DataParallel requires the forward method to be defined.
    def forward(self, data):
        # If using PyG DataParallel, it passes a list of graphs. Batch them on the current GPU.
        if isinstance(data, list):
            data = Batch.from_data_list(data)
        return self.get_loss(data)