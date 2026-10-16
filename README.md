# Generative Design of Amorphous Glassy Carbon via DDPM

### Project Overview
This project provides a generative pipeline to discover novel glassy carbon (GC) structures. By utilizing a Generative Denoising Diffusion Probabilistic Model (DDPM) paired with an $E(3)$-equivariant NequIP Graph Neural Network, the system learns the structural manifold of GC from AIMD trajectory data to sample physically valid configurations conditioned on macroscopic density.

### Repository Structure

* **Configuration**
  * `config.yaml`: Centralized hyperparameters for model architecture, diffusion schedules, and training.
  * `config_loader.py`: Utility to parse and load the YAML configuration across the pipeline.

* **Data Pipeline**
  * `data_ingestion.py`: Handles the parsing and filtering of raw AIMD `.xyz` trajectory files.
  * `dataset.py`: Converts atomic positions into Periodic Boundary Condition (PBC) aware PyTorch Geometric radius graphs.

* **Model Architecture**
  * `nequip_layer.py`: Implements the $E(3)$-equivariant tensor product layers for the GNN score-matching engine.
  * `diffusion.py`: Contains the vectorized DDPM logic, including the forward Gaussian noising schedule and reverse Langevin Dynamics sampling.

* **Execution Scripts**
  * `train.py`: Main training loop (supports multi-GPU and gradient clipping for stability).
  * `generate.py`: Inference script to sample and generate new amorphous microstructures.
  * `evaluate.py`: Physics-informed validation script that computes metrics like Radial Distribution Function (RDF) $g(r)$ peaks and dynamic density.

### Workflow & Usage

1. **Configure:** Adjust parameters in `config.yaml` (e.g., learning rate, graph cutoff radius, diffusion timesteps).
2. **Train the Model:** Run `train.py` to optimize the score-matching network on your AIMD datasets. 
3. **Generate Structures:** Use `generate.py` to denoise a random distribution of atoms into a physical GC manifold.
4. **Evaluate:** Validate the chemical short-range order and macroscopic density of your generated structures using `evaluate.py`.

### Schrödinger bridge (branch `bridge`)

`method: "bridge"` in `config.yaml` replaces the DDPM with **bridge matching from a crystal prior**. The DDPM path is unchanged (`method: "ddpm"`) and stays as the baseline.

* `priors.py`: builds the starting distribution π₀ in the target cell (`from_frames` dilated MD crystal frames, `diamond`, turbostratic `graphite`, or the uniform `gas` used by DDPM).
* `bridge.py`: pairs each glass with a prior sample, matches atoms with the Hungarian algorithm under the minimum-image distance, trains the same `NequIP_GC` to predict the remaining displacement along a Brownian bridge (`x_t = x0 + t·d + σ√(t(1−t))·ε`), and samples with the exact bridge transition in `bridge.num_sample_steps` steps (100 by default, vs 3000 for DDPM). `bridge.sigma: 0` gives deterministic flow matching.
* `evaluate.py` additionally reports coordination fractions, bond length and angle, and for the bridge the mean per-atom **travel distance from the prior** (the mode-collapse check).
* `tests/test_bridge.py`: bridge endpoints, periodic shortest paths, atom matching, and E(3)/permutation symmetry of the network. Run with `python tests/test_bridge.py`.
* `polaris/train_debug.pbs`: ALCF Polaris debug-queue job (`qsub -A <project> polaris/train_debug.pbs`); resubmit to resume from the latest checkpoint.
* Any script can use another config via `GC_CONFIG=path/to/config.yaml`.

```bash
python data_ingestion.py   # target glasses -> gc_filtered.xyz
python priors.py           # prior structures -> gc_prior.xyz
python train.py            # trains whichever `method` is set
python generate.py         # writes generated_gc.xyz and generated_gc_start.xyz
python evaluate.py
```

The current `data.raw_dataset` (gc2) is **development data only**: its amorphous half is an unquenched melt.
