import torch
from nequip_layer import NequIP_GC
from diffusion import DiffusionModel
from ase import Atoms
from ase.io import write
import numpy as np
from config_loader import load_config

config = load_config()

# GENERATION PARAMETERS
TARGET_DENSITY = config['generation']['target_density']
CELL_SIZE = config['generation']['cell_size']
NUM_ATOMS = config['generation']['num_atoms']
OUTPUT_FILENAME = config['generation']['output_filename']
BEST_CHECKPOINT = config['training']['checkpoints']['best']

ASE_ATOM_TYPES   = [6] * NUM_ATOMS      # C mass 6
MODEL_ATOM_TYPES = [0] * NUM_ATOMS      
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate():
    # Initialize models
    gnn_model = NequIP_GC(
        num_layers=config['model']['num_layers'], 
        num_types=config['model']['num_types'], 
        cutoff=config['graph']['cutoff_radius']
    )
    diffusion = DiffusionModel(
        model=gnn_model,
        num_steps=config['diffusion']['num_steps'],
        sigma_min=config['diffusion']['sigma_min'],
        sigma_max=config['diffusion']['sigma_max'],
        device=DEVICE
    )
    
    diffusion.to(DEVICE)
    
    # Load trained weights only 
    state_dict = torch.load(BEST_CHECKPOINT, map_location=DEVICE, weights_only=False)
    diffusion.load_state_dict(state_dict)
    diffusion.eval()
    print(f"Loaded weights from {BEST_CHECKPOINT}")

    # Generating loop
    print(f"Generating structure with {NUM_ATOMS} atoms...")
    generated_pos = diffusion.sample( # shape == [N,3]
        num_atoms=NUM_ATOMS,
        cell_dims=CELL_SIZE,
        atom_types=MODEL_ATOM_TYPES,
        target_density=TARGET_DENSITY
    )

    final_coords = generated_pos.cpu().numpy() # Requires nmumpy array
    
    # Creates a proper crystal structure object
    structure = Atoms( 
        symbols=ASE_ATOM_TYPES,
        positions=final_coords,
        cell=CELL_SIZE,
        pbc=True
    )

    write(OUTPUT_FILENAME, structure)
    print(f"Success! Structure saved to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    generate()