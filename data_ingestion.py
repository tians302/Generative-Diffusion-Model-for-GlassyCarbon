from ase.io import read, write
import numpy as np
from collections import Counter
from config_loader import load_config

# INGESTS MOLECULAR DATA AND CREATES ATOM OBJECTS FOR ALL NON-CRYSTALLINE MATERIALS

config = load_config()

RAW_DATASET = config['data']['raw_dataset']
ALLOWED_PHASES = config['data']['material_types']
PROCESSED_XYZ = config['data']['processed_xyz']

frames = read(RAW_DATASET, index=':')
print(f"Total frames in glassy carbon dataset: {len(frames)}")

amorphous_frames = []

for i, atoms in enumerate(frames):
    phase = atoms.info.get("phase", "unknown")

    if phase in ALLOWED_PHASES:
        atomic_numbers = atoms.get_atomic_numbers()

        # This pipeline is designed for glassy carbon only
        if not np.all(atomic_numbers == 6):
            raise ValueError(
                f"Frame {i} contains non-carbon atoms: "
                f"{sorted(set(atomic_numbers))}"
            )

        # Calculate density from atomic masses and cell volume
        mass = atoms.get_masses().sum()                 # amu
        volume = atoms.get_volume()                     # Å³
        density = mass / volume * 1.66053906660         # g/cm³

        atoms.info["density"] = float(density)
        amorphous_frames.append(atoms)

phase_counts = Counter(
    atoms.info.get("phase", "unknown") for atoms in frames
)

for phase in ALLOWED_PHASES:
    print(f"{phase:<12} : {phase_counts.get(phase, 0)}")

print(f"Total selected amorphous frames: {len(amorphous_frames)}")

if not amorphous_frames:
    raise ValueError(
        "No amorphous carbon frames were found. "
        "Check the phase labels and material_types configuration."
    )

write(PROCESSED_XYZ, amorphous_frames)
print(f"Saved filtered carbon frames to {PROCESSED_XYZ}")