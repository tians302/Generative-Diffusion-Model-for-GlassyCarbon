import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from config_loader import load_config

config = load_config()

TARGET_DENSITY = config['generation']['target_density']
R_MAX = config['graph']['cutoff_radius']
TOLERANCE = config['evaluation']['tolerance']
BINS = config['evaluation']['bins']
OUTPUT_IMG = config['evaluation']['output_img']

PROCESSED_XYZ = config['data']['processed_xyz']
GENERATED_XYZ = config['generation']['output_filename']

# Calculates the Radial Distribution Function g(r) for an ASE Atoms object.
#   1.) Draw a thin spherical shell at distance 𝑟 with thickness 𝑑𝑟 and find how many atoms are sitting in this shell
#   2.) Normalize via how many atoms would I expect in that shell if atoms were randomly distributed in space?
#   3.) g(r) = observed atoms in shell / expected atoms in shell if uniform
def calculate_rdf(atoms, r_max=R_MAX, bins=BINS):

    # Gets all pairwise distances using ASE's periodic boundary math (mic=True to handle PBC)
    distances = atoms.get_all_distances(mic=True) # distances.shape == [N, N] where distances[i, j] is the distance between atom i and atom j.
    np.fill_diagonal(distances, np.inf) # Ignores self-interactions and makes them infinity so its not shown in histogram (past cutoff point)
    distances = distances.flatten() # Flattens to [N^2]
    distances = distances[distances < r_max]

    hist, bin_edges = np.histogram(distances, bins=bins, range=(0, r_max)) # Bins the distances into a histogram
    r = (bin_edges[:-1] + bin_edges[1:]) / 2.0 # Get the center of each bin
    dr = r_max / bins # Bin width
    
    # Normalize the histogram to get the physical g(r) curve
    num_atoms = len(atoms)
    volume = atoms.get_volume()
    number_density = num_atoms / volume # Expected atoms per unit volume
    shell_volumes = 4 * np.pi * r**2 * dr # Volume of a thin spherical shell at radius r of thickness dr
    
    # g(r) = (Count in shell) / (Ideal count in shell of uniform gas = atoms * density * shell volume)
    rdf = hist / (num_atoms * number_density * shell_volumes)
    
    return r, rdf

# g/cm³ from atomic masses (amu) and cell volume (Å³)
def mass_density(atoms):
    mass = atoms.get_masses().sum()
    volume = atoms.get_volume()
    return float(mass / volume * 1.66053906660)

def frame_density(atoms):
    density = atoms.info.get("density")
    if density is None:
        return mass_density(atoms)
    return float(density)

# Plots ground truth RDF with the generated structure's RDF.
# Each curve is labeled with its own density. The generation target is
# added to the generated label when the written cell does not match it.
def plot_rdf_comparison(r, gt_rdf_avg, generated_rdf, gt_density, generated_density, target_density):
    plt.figure(figsize=(10, 6))
    plt.plot(r, gt_rdf_avg, label=f'Ground Truth (Average) - {gt_density:.2f} g/cm³', color='black', linewidth=2)

    generated_label = f'Generated Model - {generated_density:.2f} g/cm³'
    if abs(generated_density - target_density) > 0.01:
        generated_label += f' (target {target_density:.2f})'
    plt.plot(r, generated_rdf, label=generated_label, color='red', linestyle='--', linewidth=2)

    plt.xlabel('Distance r (Å)')
    plt.ylabel('Radial Distribution Function g(r)')
    plt.title('RDF Comparison: Ground Truth vs Generated Amorphous Glassy Carbon')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.savefig(OUTPUT_IMG)
    print(f"Saved to {OUTPUT_IMG}")
    return OUTPUT_IMG

def evaluate():
    print(f"--- Starting Evaluation for Target Density {TARGET_DENSITY} g/cm³ ---")

    all_structures = read(PROCESSED_XYZ, index=':') # Loads all data
    filtered_structures = [
        atoms for atoms in all_structures
        if abs(frame_density(atoms) - TARGET_DENSITY) <= TOLERANCE
    ]

    if len(filtered_structures) == 0:
        reference_density = float(np.mean([frame_density(atoms) for atoms in all_structures]))
        print(
            f"Found 0 structures within {TOLERANCE} g/cm³ of {TARGET_DENSITY}. "
            f"Using all {len(all_structures)} amorphous frames "
            f"(mean density {reference_density:.3f} g/cm³)."
        )
        filtered_structures = all_structures
    else:
        print(f"Found {len(filtered_structures)} ground truth structures matching density criteria.")

    gt_density = float(np.mean([frame_density(atoms) for atoms in filtered_structures]))

    # Calculate average RDF for the ground truth
    gt_rdf_total = np.zeros(BINS)
    for atoms in filtered_structures:
        r, rdf = calculate_rdf(atoms, r_max=R_MAX, bins=BINS)
        gt_rdf_total += rdf # Aggregates total
    gt_rdf_avg = gt_rdf_total / len(filtered_structures) # Normalizes

    generated_atoms = read(GENERATED_XYZ) # Loads generated structure
    generated_density = mass_density(generated_atoms)
    r, generated_rdf = calculate_rdf(generated_atoms, r_max=R_MAX, bins=BINS)
    print(
        f"Generated structure density is {generated_density:.3f} g/cm³ "
        f"(config target {TARGET_DENSITY} g/cm³)."
    )

    plot_rdf_comparison(
        r=r,
        gt_rdf_avg=gt_rdf_avg,
        generated_rdf=generated_rdf,
        gt_density=gt_density,
        generated_density=generated_density,
        target_density=TARGET_DENSITY,
    )

if __name__ == "__main__":
    evaluate()