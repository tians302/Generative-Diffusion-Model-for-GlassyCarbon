from ase.io import read, write
import numpy as np
from collections import Counter
from config_loader import load_config

# INGESTS MOLECULAR DATA AND CREATES ATOM OBJECTS FOR ALL NON-CRYSTALLINE MATERIALS

config = load_config()

RAW_DATASET = config['data']['raw_dataset']
MATERIAL_TYPES = config['data']['material_types']
PROCESSED_XYZ = config['data']['processed_xyz']

frames = read(RAW_DATASET, index=':')
print(f"Total frames in SiC dataset: {len(frames)}")

amorphous_frames = []

for i, atoms in enumerate(frames):
    config_type = atoms.info.get('config_type', 'unknown') # Default is unknown if there is no config_type mentioned

    if config_type in MATERIAL_TYPES:
        
        # Extracted quantities
        elements = atoms.get_chemical_symbols()
        atomic_numbers = atoms.numbers
        material_type = atoms.info["config_type"]
        mass = atoms.get_masses().sum()                    # amu
        volume = atoms.get_volume()                        # Å^3
        density_calc = (mass / volume) * 1.66053906660     # g/cm^3
        density = atoms.info.get("density", density_calc)
        atoms.info["density"] = float(density)
        forces = atoms.get_forces()                         # eV/Å
        energy = atoms.get_potential_energy()               # eV
        free_energy = atoms.calc.results.get("free_energy") # eV
        stress = atoms.get_stress(voigt=False)              # eV/Å^3 (3x3)
        pbc = atoms.pbc                                     # (3,) bool

        # print("elements:", elements)
        # print("atomic_numbers:", atomic_numbers)
        # print("material_type:", material_type)
        # print("mass (amu):", mass, " total:", mass.sum())
        # print("volume (Å^3):", volume)
        # print("density (g/cm^3):", density)
        # print("forces (eV/Å) shape:", forces.shape, " first row:", forces[0])
        # print("energy (eV):", energy)
        # print("free_energy (eV):", free_energy)
        # print("stress (eV/Å^3) shape:", stress.shape)
        # print(stress)
        # print("pbc:", pbc)

        amorphous_frames.append(atoms)

config_counts = Counter(atoms.info.get("config_type", "unknown") for atoms in frames)
for t in MATERIAL_TYPES:
    print(f"{t:<12} : {config_counts.get(t, 0)}")
print(f"Total amorphous solids: {len(amorphous_frames)}")

write(PROCESSED_XYZ, amorphous_frames)
print(f"Saved to {PROCESSED_XYZ}")