import os
import torch
from nequip_layer import NequIP_GC
from diffusion import DiffusionModel
from bridge import BridgeModel, load_prior_positions, make_prior_batch
from ase import Atoms
from ase.io import write
import numpy as np
from config_loader import load_config

config = load_config()

# GENERATION PARAMETERS
METHOD = config.get('method', 'ddpm') # "ddpm" or "bridge"
TARGET_DENSITY = config['generation']['target_density']
CELL_SIZE = config['generation']['cell_size']
NUM_ATOMS = config['generation']['num_atoms']
NUM_STRUCTURES = config['generation'].get('num_structures', 1)
OUTPUT_FILENAME = config['generation']['output_filename']
START_FILENAME = config['generation'].get('start_filename', 'generated_gc_start.xyz')
BEST_CHECKPOINT = config['bridge' if METHOD == 'bridge' else 'diffusion']['checkpoints']['best']

ASE_ATOM_TYPES   = [6] * NUM_ATOMS      # C mass 6
MODEL_ATOM_TYPES = [0] * NUM_ATOMS      
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_atoms(positions):
    # Creates a proper crystal structure object
    return Atoms(symbols=ASE_ATOM_TYPES, positions=positions, cell=CELL_SIZE, pbc=True)

def generate():
    # Initialize models
    gnn_model = NequIP_GC(
        num_layers=config['model']['num_layers'], 
        num_types=config['model']['num_types'], 
        cutoff=config['graph']['cutoff_radius']
    )
    if METHOD == 'bridge':
        diffusion = BridgeModel(
            model=gnn_model,
            sigma=config['bridge']['sigma'],
            num_sample_steps=config['bridge']['num_sample_steps'],
            device=DEVICE
        )
    else:
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
    print(f"Generating {NUM_STRUCTURES if METHOD == 'bridge' else 1} structure(s) with {NUM_ATOMS} atoms via {METHOD}...")
    if METHOD == 'bridge':
        # Start each structure from a random prior sample (shifted at random inside make_prior_batch)
        prior_positions = load_prior_positions(config['bridge']['prior_xyz'])
        if prior_positions.shape[1] != NUM_ATOMS:
            raise ValueError(f"Prior has {prior_positions.shape[1]} atoms but generation.num_atoms = {NUM_ATOMS}")
        picks = torch.randint(len(prior_positions), (NUM_STRUCTURES,))
        prior_batch = make_prior_batch([prior_positions[i] for i in picks], CELL_SIZE, MODEL_ATOM_TYPES, TARGET_DENSITY)
        start = prior_batch.pos.clone()
        generated_pos = diffusion.sample(prior_batch).cpu() # shape == [NUM_STRUCTURES * N, 3]

        # Same atom order in both files, so evaluate.py can measure how far each atom travelled
        structures = [to_atoms(p.numpy()) for p in generated_pos.split(NUM_ATOMS)]
        starts = [to_atoms(p.numpy()) for p in start.split(NUM_ATOMS)]
        write(START_FILENAME, starts)
        print(f"Saved starting priors to {START_FILENAME}")
    else:
        generated_pos = diffusion.sample( # shape == [N,3]
            num_atoms=NUM_ATOMS,
            cell_dims=CELL_SIZE,
            atom_types=MODEL_ATOM_TYPES,
            target_density=TARGET_DENSITY
        )
        structures = [to_atoms(generated_pos.cpu().numpy())] # Requires nmumpy array

    write(OUTPUT_FILENAME, structures)
    print(f"Success! {len(structures)} structure(s) saved to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    generate()
