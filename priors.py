import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read, write
from config_loader import load_config

# BUILDS THE STARTING DISTRIBUTION (pi_0) FOR THE SCHRODINGER BRIDGE
#   The bridge transports a prior structure into a glass, so every prior sample must live in the
#   SAME space as the targets: same number of atoms, same species, same cell.
#   Prior kinds:
#       from_frames --> crystal frames from an MD file (e.g. thermal diamond), dilated into the target cell
#       diamond     --> ideal diamond supercell dilated into the target cell (+ optional thermal jitter)
#       graphite    --> graphene sheets with in-plane C-C kept at 1.42 A, interlayer spacing set by the density
#       gas         --> uniform random positions (the DDPM prior, kept for the prior ablation)

config = load_config()

C_MASS = 12.011
AMU_PER_A3_TO_GCC = 1.66053906660
DIAMOND_A = 3.567   # Å, cubic diamond lattice constant
CC_BOND = 1.42      # Å, graphene in-plane bond length


def dilate_to_cell(atoms, cell_lengths):
    # Keeps fractional coordinates fixed and stretches the box --> topology is preserved, bonds are rescaled
    scaled = atoms.get_scaled_positions(wrap=True)
    new = Atoms(symbols=atoms.get_chemical_symbols(), cell=np.diag(cell_lengths), pbc=True)
    new.set_scaled_positions(scaled)
    return new


def jitter(atoms, sigma, rng):
    # Small Gaussian displacements so an ideal lattice prior is a distribution, not a single point
    if sigma > 0:
        atoms.positions += rng.normal(scale=sigma, size=atoms.positions.shape)
        atoms.wrap()
    return atoms


def diamond_prior(num_atoms, cell_lengths, jitter_sigma, rng):
    # 8 atoms per cubic diamond cell --> num_atoms must be 8 * n^3 (216 = 3x3x3, 1000 = 5x5x5)
    n = round((num_atoms / 8) ** (1 / 3))
    if 8 * n ** 3 != num_atoms:
        raise ValueError(f"Diamond prior needs num_atoms = 8 n^3, got {num_atoms}")
    atoms = bulk("C", "diamond", a=DIAMOND_A, cubic=True).repeat((n, n, n))
    return jitter(dilate_to_cell(atoms, cell_lengths), jitter_sigma, rng)


def graphite_prior(num_atoms, cell_lengths, jitter_sigma, rng):
    # Rectangular graphene cell: a = sqrt(3) * a_cc along x, b = 3 * a_cc along y, 4 atoms per layer per cell
    a = np.sqrt(3) * CC_BOND
    b = 3 * CC_BOND
    basis = np.array([[0, 0], [a / 2, CC_BOND / 2], [a / 2, 1.5 * CC_BOND], [0, 2 * CC_BOND]])

    Lx, Ly, Lz = cell_lengths
    nx, ny = round(Lx / a), round(Ly / b)
    if not (np.isclose(nx * a, Lx, rtol=1e-3) and np.isclose(ny * b, Ly, rtol=1e-3)):
        raise ValueError(
            f"Graphite prior needs Lx = n*{a:.4f} and Ly = m*{b:.4f} Å; got {Lx}, {Ly}. "
            f"For 1000 atoms at 1.5 g/cc use cell [24.596, 21.30, 25.371]."
        )
    per_layer = 4 * nx * ny
    nz = num_atoms // per_layer
    if nz * per_layer != num_atoms:
        raise ValueError(f"num_atoms={num_atoms} is not a whole number of {per_layer}-atom layers")

    gx, gy = np.meshgrid(np.arange(nx) * a, np.arange(ny) * b, indexing="ij")
    origins = np.stack([gx.ravel(), gy.ravel()], axis=1)
    layer = (origins[:, None, :] + basis[None, :, :]).reshape(-1, 2)

    # Turbostratic stacking: every sheet gets a random in-plane shift, so the prior has real variety
    d = Lz / nz
    positions = []
    for k in range(nz):
        shift = rng.uniform(0, 1, size=2) * [Lx, Ly]
        xy = layer + shift
        positions.append(np.column_stack([xy, np.full(len(xy), k * d)]))

    atoms = Atoms("C" * num_atoms, positions=np.concatenate(positions), cell=np.diag(cell_lengths), pbc=True)
    atoms.wrap()
    return jitter(atoms, jitter_sigma, rng)


def gas_prior(num_atoms, cell_lengths, rng):
    return Atoms("C" * num_atoms, scaled_positions=rng.uniform(size=(num_atoms, 3)),
                 cell=np.diag(cell_lengths), pbc=True)


def build_priors(kind, num_atoms, cell_lengths, num_samples, source_xyz=None, source_phase="crystal",
                 jitter_sigma=0.05, seed=0):
    rng = np.random.default_rng(seed)
    cell_lengths = np.asarray(cell_lengths, dtype=float)

    if kind == "from_frames":
        frames = [a for a in read(source_xyz, index=":") if a.info.get("phase") == source_phase]
        if not frames:
            raise ValueError(f"No frames with phase={source_phase} in {source_xyz}")
        if any(len(a) != num_atoms for a in frames):
            raise ValueError(f"Prior frames must have {num_atoms} atoms")
        priors = [dilate_to_cell(a, cell_lengths) for a in frames]
    elif kind == "diamond":
        priors = [diamond_prior(num_atoms, cell_lengths, jitter_sigma, rng) for _ in range(num_samples)]
    elif kind == "graphite":
        priors = [graphite_prior(num_atoms, cell_lengths, jitter_sigma, rng) for _ in range(num_samples)]
    elif kind == "gas":
        priors = [gas_prior(num_atoms, cell_lengths, rng) for _ in range(num_samples)]
    else:
        raise ValueError(f"Unknown prior kind: {kind}")

    for a in priors:
        a.info["prior_kind"] = kind
        a.info["density"] = float(len(a) * C_MASS / a.get_volume() * AMU_PER_A3_TO_GCC)
    return priors


def main():
    bridge_cfg = config["bridge"]
    gen_cfg = config["generation"]
    priors = build_priors(
        kind=bridge_cfg["prior_kind"],
        num_atoms=gen_cfg["num_atoms"],
        cell_lengths=gen_cfg["cell_size"],
        num_samples=bridge_cfg["num_prior_samples"],
        source_xyz=config["data"]["raw_dataset"],
        source_phase=bridge_cfg["prior_source_phase"],
        jitter_sigma=bridge_cfg["prior_jitter"],
        seed=bridge_cfg["seed"],
    )
    write(bridge_cfg["prior_xyz"], priors)
    print(f"Saved {len(priors)} '{bridge_cfg['prior_kind']}' prior structures to {bridge_cfg['prior_xyz']} "
          f"({len(priors[0])} atoms, cell {priors[0].cell.lengths()}, {priors[0].info['density']:.3f} g/cm³)")


if __name__ == "__main__":
    main()
