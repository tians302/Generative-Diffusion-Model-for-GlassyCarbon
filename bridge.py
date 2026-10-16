import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from config_loader import load_config
from diffusion import build_pbc_graph

# SCHRODINGER BRIDGE (BRIDGE MATCHING) FROM A CRYSTAL PRIOR TO GLASSY CARBON
#   DDPM starts from a uniform gas; here every structure starts from a prior sample x0 (a crystal) and is
#   transported to a glass x1 along a Brownian bridge with reference noise sigma:
#       x_t = x0 + t * d + sigma * sqrt(t (1 - t)) * eps,      d = minimum_image(x1 - x0)
#   The network (the same NequIP_GC used by DDPM) predicts the remaining displacement r = x1 - x_t, which is
#   still one equivariant 3-vector per atom. Sampling follows the bridge toward the predicted endpoint.
#   Coupling: prior and target are paired at random, then atoms are matched with the Hungarian algorithm under
#   the minimum-image distance (after a random global shift of the prior), so transport paths are short.
#   Fixed coupling = one step of Iterative Markovian Fitting (bridge matching); it is not yet the full IMF/DSBM SB.

config = load_config()

CUTOFF_RADIUS = config['graph']['cutoff_radius']
MAX_NEIGHBORS = config['graph']['max_neighbors']
SIGMA = config['bridge']['sigma']
NUM_SAMPLE_STEPS = config['bridge']['num_sample_steps']
SHIFT_TRIALS = config['bridge']['shift_trials']


def minimum_image(vec, cell):
    return vec - torch.round(vec / cell) * cell


def wrap(x, cell):
    return x - torch.floor(x / cell) * cell


# Pairs every target atom with one prior atom so the total squared minimum-image displacement is smallest
def match_atoms(x0, x1, cell, shift_trials=SHIFT_TRIALS, random_shift=True):
    # x0, x1 shape: [N, 3], cell shape: [3]
    best_cost, best_x0 = float('inf'), None
    for _ in range(shift_trials if random_shift else 1):
        # The prior is translation invariant under PBC, so a random global shift is a free augmentation
        shift = torch.rand(3) * cell if random_shift else torch.zeros(3)
        x0_shifted = wrap(x0 + shift, cell)
        diff = minimum_image(x1[:, None, :] - x0_shifted[None, :, :], cell) # [N, N, 3]
        cost = (diff ** 2).sum(-1).numpy() # [N, N]
        rows, cols = linear_sum_assignment(cost) # rows == arange(N); cols[i] is the prior atom paired with target atom i
        total = cost[rows, cols].sum()
        if total < best_cost:
            best_cost, best_x0 = total, x0_shifted[torch.as_tensor(cols)]
    return best_x0, best_cost / len(x1) # matched prior [N, 3], mean squared displacement (Å²)


# Wraps the glass dataset: each item is a glass x1 plus a matched prior sample x0 (stored as pos0)
class BridgePairDataset(Dataset):
    def __init__(self, targets, priors, shift_trials=SHIFT_TRIALS):
        self.targets = targets # GCGraphDataset or a Subset of it
        self.priors = priors # [P, N, 3], already in the target cell
        self.shift_trials = shift_trials

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        g = self.targets[idx]
        cell = g.cell.reshape(3)
        if g.pos.shape[0] != self.priors.shape[1]:
            raise ValueError(f"Target has {g.pos.shape[0]} atoms but priors have {self.priors.shape[1]}")

        # torch RNG (not numpy) so every DataLoader worker draws different priors and shifts
        x0 = self.priors[torch.randint(len(self.priors), (1,)).item()]
        x1 = wrap(g.pos, cell)
        x0_matched, _ = match_atoms(x0, x1, cell, self.shift_trials)
        return Data(pos=x1, pos0=x0_matched, z=g.z, y=g.y, cell=g.cell)


class BridgeModel(nn.Module):
    def __init__(self, model, sigma=SIGMA, num_sample_steps=NUM_SAMPLE_STEPS, device='cpu'):
        super().__init__()
        self.model = model.to(device)
        self.sigma = sigma
        self.num_sample_steps = num_sample_steps
        self.device = device

    # Builds PBC edges graph by graph and offsets node indices into the concatenated batch
    def build_edges(self, pos, batch, cell, num_graphs):
        edge_indices = []
        offset = 0
        for i in range(num_graphs):
            pos_i = pos[batch == i]
            edges_i = build_pbc_graph(pos_i, cell[i], cutoff=CUTOFF_RADIUS, max_neighbors=MAX_NEIGHBORS)
            edge_indices.append(edges_i + offset)
            offset += pos_i.shape[0]
        return torch.cat(edge_indices, dim=1) # [2, E_total]

    # Samples a point on the Brownian bridge between x0 and x1 and returns the displacement still to travel
    def q_sample(self, x0, x1, t_atom, cell_atom):
        # x0, x1, cell_atom shape: [N_total, 3], t_atom shape: [N_total, 1]
        d = minimum_image(x1 - x0, cell_atom) # Shortest periodic path from prior atom to its matched glass atom
        std = self.sigma * torch.sqrt(t_atom * (1 - t_atom))
        noise = torch.randn_like(x0) * std
        x_t = wrap(x0 + t_atom * d + noise, cell_atom)
        remaining = (1 - t_atom) * d - noise # == minimum_image(x1 - x_t), written analytically so no wrap ambiguity
        return x_t, remaining

    def get_loss(self, data):
        device = data.pos.device
        batch_size = data.num_graphs
        cell = data.cell.reshape(-1, 3) # [Batch, 3]

        # One continuous bridge time per graph, t ~ U(0, 1)
        t = torch.rand(batch_size, device=device)
        t_atom = t[data.batch][:, None]
        x_t, target = self.q_sample(data.pos0, data.pos, t_atom, cell[data.batch])

        noisy_data = data.clone()
        noisy_data.pos = x_t
        noisy_data.edge_index = self.build_edges(x_t, data.batch, cell, batch_size)

        predicted = self.model(noisy_data, t) # [N_total, 3]
        return F.mse_loss(predicted, target)

    # Transports prior structures to glasses; prior_batch holds pos (prior), z, y (target density), cell, batch
    @torch.no_grad()
    def sample(self, prior_batch, num_steps=None, return_trajectory=False):
        num_steps = num_steps or self.num_sample_steps
        data = prior_batch.to(self.device)
        batch_size = data.num_graphs
        cell = data.cell.reshape(-1, 3)
        cell_atom = cell[data.batch]

        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)
        x = wrap(data.pos, cell_atom)
        trajectory = [x.clone()] if return_trajectory else None

        for k in tqdm(range(num_steps)):
            t, t_next = ts[k], ts[k + 1]
            step = data.clone()
            step.pos = x
            step.edge_index = self.build_edges(x, data.batch, cell, batch_size)
            remaining = self.model(step, t.expand(batch_size)) # Predicted x1 - x_t

            # Exact Brownian-bridge transition from (x, t) toward the predicted endpoint x + remaining.
            # The noise vanishes at t_next = 1, so the last step lands exactly on the predicted glass.
            dt = t_next - t
            mean = x + (dt / (1 - t)) * remaining
            std = self.sigma * torch.sqrt(dt * (1 - t_next) / (1 - t))
            x = wrap(mean + std * torch.randn_like(x), cell_atom)
            if return_trajectory:
                trajectory.append(x.clone())

        return (x, trajectory) if return_trajectory else x

    # PyTorch DataParallel requires the forward method to be defined.
    def forward(self, data):
        if isinstance(data, list):
            data = Batch.from_data_list(data)
        return self.get_loss(data)


# Reads the prior file written by priors.py into a [P, N, 3] tensor
def load_prior_positions(prior_xyz):
    from ase.io import read
    frames = read(prior_xyz, index=':')
    return torch.stack([torch.tensor(a.get_positions(wrap=True), dtype=torch.float32) for a in frames])


# Turns a list of prior position tensors into a batch ready for BridgeModel.sample
def make_prior_batch(prior_positions, cell_dims, atom_types, target_density):
    cell = torch.tensor(cell_dims, dtype=torch.float32)
    graphs = []
    for x0 in prior_positions:
        x0 = wrap(x0 + torch.rand(3) * cell, cell) # Random global shift, as during training
        graphs.append(Data(
            pos=x0,
            z=torch.tensor(atom_types, dtype=torch.long),
            y=torch.tensor([target_density], dtype=torch.float32),
            cell=cell.unsqueeze(0),
        ))
    return Batch.from_data_list(graphs)
