# Generative Design of Amorphous SiC via DDPM

### Project Overview
This project provides a generative pipeline to discover novel amorphous Silicon Carbide (SiC) microstructures for high-temperature aerospace coatings. By utilizing a Generative Denoising Diffusion Probabilistic Model (DDPM) paired with an $E(3)$-equivariant NequIP Graph Neural Network, the system learns the structural manifold of SiC from AIMD trajectory data to sample physically valid configurations conditioned on macroscopic density.

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
  * `generate.py`: Inference script to sample and generate new amorphous SiC microstructures.
  * `evaluate.py`: Physics-informed validation script that computes metrics like Radial Distribution Function (RDF) $g(r)$ peaks and dynamic density.

### Workflow & Usage

1. **Configure:** Adjust parameters in `config.yaml` (e.g., learning rate, graph cutoff radius, diffusion timesteps).
2. **Train the Model:** Run `train.py` to optimize the score-matching network on your AIMD datasets. 
3. **Generate Structures:** Use `generate.py` to denoise a random distribution of atoms into a physical SiC manifold.
4. **Evaluate:** Validate the chemical short-range order and macroscopic density of your generated structures using `evaluate.py`.