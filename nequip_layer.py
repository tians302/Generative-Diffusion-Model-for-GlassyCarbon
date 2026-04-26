import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import Gate
from torch_geometric.utils import scatter
from config_loader import load_config
import warnings

config = load_config()
warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support instance-level annotations")

NUM_LAYERS = config['model']['num_layers']
NUM_TYPES = config['model']['num_types']
EMBEDDING_DIM = config['model']['embedding_dim']
BESSEL_BASIS = config['model']['bessel_basis']
IRREPS_NODE = config['model']['irreps_node']
IRREPS_SH = config['model']['irreps_sh']
IRREPS_EDGE = config['model']['irreps_edge']
DENSITY_MIN = config['model']['density_min']
DENSITY_MAX = config['model']['density_max']
CUTOFF_RADIUS = config['graph']['cutoff_radius']
MAX_NEIGHBORS = config['graph']['max_neighbors']


# It converts a single number into a vector of size 64 so it can be added to atom features.
class ScalarEmbedding(nn.Module):
    def __init__(self, num_basis=EMBEDDING_DIM, hidden_dim=64, min_val=0.0, max_val=1.0): # Paper had cooling rate embedding num_basis at 9, but left this as 8 for generality
        super().__init__()
        # Creates num_basis evenly spaced centers between the min and max value
        self.register_buffer("centers", torch.linspace(min_val, max_val, num_basis)) # register_buffer is Pytorch semantic for this
        self.sigma = (max_val - min_val) / (num_basis - 1) # Computes spacing between centers

        # MLP to project the Gaussian bumps into a 64-dimensional feature vector
        self.mlp = nn.Sequential(
            nn.Linear(num_basis, 64), # Projects to 64
            nn.Softplus(), # Adds non-linearity
            nn.Linear(64, hidden_dim)
        )

    def forward(self, scalar_value):
        if scalar_value.dim() == 1:
            scalar_value = scalar_value.unsqueeze(-1) # Changes from [1] to [1,1] to be able to subtract the centers
        
        # Turns distances into Gaussian bumps ( if scalar value is on center, basis --> 1 where far from center, basis --> 0)
        basis = torch.exp(-(scalar_value - self.centers)**2 / (2 * self.sigma**2)) # Gaussian formula
        return self.mlp(basis) # Turns into a feature vector of shape [B,64]

# Instead of using a single scalar quantity to represent distance, use a feature vector with non-linearity to map complex relationship.
# φ_n(r) ∝ sqrt(2 / r_c) * sin(nπ r / r_c) / r and set φ_n(r) = 0 for r >= r_c (hard cutoff).
class BesselBasis(nn.Module):
    def __init__(self, cutoff=CUTOFF_RADIUS, num_basis=BESSEL_BASIS):
        super().__init__()
        self.cutoff = cutoff
        self.num_basis = num_basis # Length of feature vector

        # Precompute frequencies [pi, 2pi... 8pi]
        self.register_buffer("freq", torch.pi * torch.arange(1, num_basis + 1)) 

    def forward(self, dist):
        x = dist / self.cutoff 
        # Equation: 1 - 6x^5 + 15x^4 - 10x^3 --> ensures the value and its derivatives are exactly 0 at the cutoff
        envelope = 1.0 - 6.0 * x**5 + 15.0 * x**4 - 10.0 * x**3
        bessel = (2.0 / self.cutoff)**0.5 * torch.sin(self.freq * x) / (x + 1e-8) # Adds a small epsilon to dist to avoid division by zero at r=0
        out = envelope * bessel
        # torch.where(condition, A, B)
        return torch.where(dist < self.cutoff, out, torch.zeros_like(out))

# It computes the interaction between atoms based on their distance and relative orientation (spherical harmonics). Where the message passing happens
class InteractionBlock(nn.Module):
    def __init__(self, irreps_node, irreps_node_out, irreps_edge, number_of_basis=BESSEL_BASIS, avg_num_neighbors=MAX_NEIGHBORS):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        
        # Combines Node Features + Edge Features -> Output Node Features while preserving equivariance via CBTP
        self.tp = FullyConnectedTensorProduct(
            irreps_in1=irreps_node,
            irreps_in2=irreps_edge,
            irreps_out=irreps_node_out,
            internal_weights=False, # We generate weights dynamically from distance
            shared_weights=False # Want edges to have their own distance dependent weights  
        )
        
        # Radial Network: Input (8 basis) -> Linear(16) -> SiLU -> Linear(64) -> SiLU -> Linear(Weights)
        self.fc = nn.Sequential(
            nn.Linear(number_of_basis, 16), # Input shape [E,8] --> [E,16]
            nn.SiLU(), # Shape remains [E,16] --> adds non-linearity to each argument
            nn.Linear(16, 64), # [E,16] --> [E,64]
            nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel) # Output size matches what TP needs
        )
        
        # Residual Connection (Self-Interaction) --> updates atom using only its current features (no edges)
        self.sc = o3.Linear(irreps_node, irreps_node_out)

    def forward(self, node_features, edge_features, edge_dist_bessel, edge_index):
        # Computes distance-dependent weights
        weight = self.fc(edge_dist_bessel) # [E,8] --> [E,weight_numel]
        
        # Message Passing
        neighbor_nodes = node_features[edge_index[1]] # For each edge, grab the features of the neighbor nodes (ones where it has edges connected)
        messages = self.tp(neighbor_nodes, edge_features, weight) # Tensor product geometry
        node_output = scatter(messages, edge_index[0], dim=0, reduce='add') # Adds neighboring info [N, irreps_node_out]
        node_output = node_output / (self.avg_num_neighbors ** 0.5)
        return node_output + self.sc(node_features) # Adds self-interaction block
    
# Enriches edge embeddings from "1x0e + 1x1o + 1x2e" to "4x0e + 4x1o + 2x2e" by mixing scalar distances with directional geometry
class RichEdgeEmbedder(nn.Module):
    def __init__(self, bessel_dim=BESSEL_BASIS, irreps_sh=IRREPS_SH, irreps_edge_out=IRREPS_EDGE):
        super().__init__()
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_edge_out = o3.Irreps(irreps_edge_out)
        
        # Tensor product: takes "1x0e + 1x1o + 1x2e" and multiplies each by 1.0 (doing nothing) and then multiplies by 10 dynamic weights by the MLP
        self.tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_sh,    # [E, 1x0e + 1x1o + 1x2e]
            irreps_in2="1x0e",            # [E, 1] (Dummy multiplier)
            irreps_out=self.irreps_edge_out, # [E, 4x0e + 4x1o + 2x2e]
            internal_weights=False,       # MLP generates weights
            shared_weights=False          # Each edge gets its own weights based on distance
        )
        
        # Radial MLP: Converts 8 Bessel scalars into exactly 10 weights needed by the TP
        self.mlp = nn.Sequential(
            nn.Linear(bessel_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel) 
        )

    def forward(self, edge_bessel, edge_sh):
        weights = self.mlp(edge_bessel) # [E, 8] --> [E, 10]
        dummy_in2 = torch.ones(edge_bessel.shape[0], 1, device=edge_bessel.device) # [E, 1]
        
        # Returns enriched edges of shape [E, 4x0e + 4x1o + 2x2e]
        rich_edge_features = self.tp(edge_sh, dummy_in2, weights) 
        return rich_edge_features

# Translation invariant due to using relative atomic distances of atoms
# Permuation invariant (swapping atoms lables makes the feature vector move with it; total energy is invariant to permutation; message passing layers summing information from neighboring atoms (order does not matter)
# Rotation invariant due to spherical harmonics --> Sm(l)​(rij​)=R(rij​)Ym(l)​(r^ij​) --> angular part is hard coded symmetry and the only learnable weights are from R(r) which determines how strongly to weight interactions at each distances
class NequIP_SiC(nn.Module):
    def __init__(self, num_layers=NUM_LAYERS, num_types=NUM_TYPES, cutoff=CUTOFF_RADIUS):
        super().__init__()
        self.cutoff = cutoff

        self.irreps_node = o3.Irreps(IRREPS_NODE) # Node feature representation with 64 scalar and 43 vector channels
        self.irreps_sh = o3.Irreps(IRREPS_SH) # Raw edge geometric directions
        self.irreps_edge = o3.Irreps(IRREPS_EDGE) # Enriched edge feature representation
        
        self.edge_enricher = RichEdgeEmbedder(
            bessel_dim=BESSEL_BASIS, 
            irreps_sh=self.irreps_sh, 
            irreps_edge_out=self.irreps_edge
        )

        # Maps atomic species onto an 8-dimensional scalar feature space 
        # Ex. z = [0,1,1,0], self.embedding(z).shape == [4,8]
        self.embedding_dim = EMBEDDING_DIM
        self.embedding = nn.Embedding(num_types, self.embedding_dim) # Lookup table --> ex. every C atom gets the same learned 8 numbers (until training updates them)
        self.irreps_atom_type = o3.Irreps(f"{self.embedding_dim}x0e") # Atom type (element) feature representation with 8 scalar channels
        # Projects atom_type representation to same as the nodes (64 scalars + 32 vectors) --> vectors are padded with zeros initially, effectively mapping scalars from 8 --> 64
        self.atom_type_embedding = o3.Linear(self.irreps_atom_type, self.irreps_node)

        # Coditional Embeddings
        # Density Embedding --> handles range 2.0 to 3.5 g/cm^3 (Typical SiC)
        self.density_embedder = ScalarEmbedding(num_basis=EMBEDDING_DIM, hidden_dim=64, min_val=DENSITY_MIN, max_val=DENSITY_MAX)
        # Time Embedder --> handles steps 0 to 1000 (Standard Diffusion)
        self.time_embedder = ScalarEmbedding(num_basis=EMBEDDING_DIM, hidden_dim=64, min_val=0, max_val=1.0)

        # Radial Basis --> converts edge distances (r_ij) to a 8-number feature vector (answers how strong should neighbor j influence atom i)
        self.radial_basis = BesselBasis(cutoff=cutoff, num_basis=BESSEL_BASIS) 

        # Gating --> applies non-linearity to vectors by multiplying the entire vector by a scalar gate
        irreps_scalars = o3.Irreps("64x0e")
        irreps_scalar_gates = o3.Irreps("32x0e") # Required for rotational equivariance
        irreps_gated_vectors   = o3.Irreps("32x1o")
        self.irreps_iblock_out = irreps_scalars + irreps_scalar_gates + irreps_gated_vectors # The InteractionBlock outputs all of these combined --> 96x0e + 32x1o
        
        self.layers = nn.ModuleList() # Pytorch list
        for _ in range(num_layers): # "num_convs = 3"
            block = InteractionBlock(
                irreps_node=self.irreps_node, # 64x0e + 32x1o
                irreps_node_out=self.irreps_iblock_out, # 96x0e + 32x1o
                irreps_edge=self.irreps_edge, # 1x0e + 1x1o + 1x2e
                number_of_basis=BESSEL_BASIS
            )
            # The Gate applies Non-Linearity (gate.shape == 64x0e + 32x1o)
            gate = Gate(
                irreps_scalars, [torch.nn.SiLU()], # Applied SiLu to 64 scalars
                irreps_scalar_gates,   [torch.nn.Sigmoid()], # Applied sigmoid to 32 scalars, becoming [0-1]
                irreps_gated_vectors # Multiplies each vector by the scalar gate and then drops gate scalar
            )
            self.layers.append(nn.ModuleList([block, gate]))
            
        self.final_layer = o3.Linear(self.irreps_node, "1x1o") # Final output of predicted noise (1 vector per atom)

    def forward(self, data, timestep):
        pos = data.pos # pos.shape == [N,3]
        edge_index = data.edge_index # edge_index.shape == [2,E]
        z = data.z # z.shape == [N]
        density = data.y # density.shape == [1]
        batch = data.batch if hasattr(data, 'batch') else torch.zeros_like(z) # Encodes which atoms are in which graph --> batch = tensor([0, 0, 0, 1, 1]) which should 2 graphs
        
        # Computes edge vector --> r_ij = x_j - x_i fpr each edge
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]] # edgevec.shape == [E,3]
        edge_dist = edge_vec.norm(dim=1, keepdim=True) # Normalizes to [E]
        
        # Converts each r_ij vector into a set of spherical harmonic features and encodes distances
        edge_sh = o3.spherical_harmonics( # edge_sh.shape == [E, 26]
            self.irreps_sh, edge_vec, normalize=True, normalization='component'
        )
        edge_bessel = self.radial_basis(edge_dist) # edge_bessel.shape == [E,8] (calls forward method of class)
        rich_edge_features = self.edge_enricher(edge_bessel, edge_sh) # rich_edge_features.shape == [E, 4x0e + 4x1o + 2x2e]
        
        # Embeds atom type
        h = self.embedding(z)            # [N, 8]
        h = self.atom_type_embedding(h)  # [N, 64 scalars + 32 vectors (zeros)]

        # Gets vectors for Density and Time
        d_vec = self.density_embedder(density)   # [Batch, 64]
        t_vec = self.time_embedder(timestep)     # [Batch, 64]
        global_cond = d_vec + t_vec              # [Batch, 64] --> Condition Vector = Density + Time
        
        # Add to every atom (Broadcast using batch index) --> only adds to the first 64 columns (the scalars)
        h_scalars = h[:, :64] + global_cond[batch] # copies the right graph conditioning to each atom
        h_vectors = h[:, 64:]
        h = torch.cat([h_scalars, h_vectors], dim=1)
        
        # Message Passing Layers
        for block, gate in self.layers:
            h_expanded = block(h, rich_edge_features, edge_bessel, edge_index) # Expands to include gates (64x0e + 32x0e + 32x1o)
            h = gate(h_expanded) # Contracts back to irreps_node (64x0e + 32x1o)
            
        noise_pred = self.final_layer(h) # 64x0e + 32x1o → 1x1o to predict the noise
        return noise_pred

if __name__ == "__main__":
    model = NequIP_SiC()
    print("Model Built Successfully to Paper Specs!")
    print(f"Hidden: {model.irreps_node}")
    print(f"Edges:  {model.irreps_edge}")
    print(f"Params: {sum(p.numel() for p in model.parameters())}")