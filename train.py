import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from dataset import SiCGraphDataset
from nequip_layer import NequIP_SiC
from diffusion import DiffusionModel
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel as PyG_DataParallel
import os
from config_loader import load_config
import wandb
import torch.profiler

config = load_config()

# TRAINING HYPERPARAMETERS
BATCH_SIZE = config['training']['batch_size']
LEARNING_RATE = float(config['training']['learning_rate'])
NUM_EPOCHS = config['training']['num_epochs']
VAL_SPLIT = config['training']['val_split']
LOG_INTERVAL = config['training']['log_interval']

PROCESSED_XYZ = config['data']['processed_xyz']
LATEST_CHECKPOINT = config['training']['checkpoints']['latest']
BEST_CHECKPOINT = config['training']['checkpoints']['best']
FINAL_CHECKPOINT = config['training']['checkpoints']['final']
LOSS_CURVE_IMG = config['training']['loss_curve_img']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    # WANDB intialization
    wandb.init(
        project="sic-diffusion-benchmark",
        config=config, # Automatically logs your hyperparameters!
        name="multi-gpu-benchmark"
    )

    if not os.path.exists(PROCESSED_XYZ):
        print(f"Dataset '{PROCESSED_XYZ}' not found. Running data_ingestion.py...")
        import data_ingestion

    # Loads and splits dataset
    dataset = SiCGraphDataset(root='.', xyz_file=PROCESSED_XYZ)
    dataset_size = len(dataset)
    val_size = max(1, int(dataset_size * VAL_SPLIT))
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataListLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataListLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # Initializing models
    gnn_model = NequIP_SiC(
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
    
    # DataParallel wrapper logic:
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via PyG DataParallel!")
        diffusion = PyG_DataParallel(diffusion)

    diffusion.to(DEVICE) # Keep this to push the base model
    
    optimizer = optim.Adam(diffusion.parameters(), lr=LEARNING_RATE)
    
    # Tracking Variables & Resume Logic
    start_epoch = 1
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') # Start with infinity so the first epoch always saves
    latest_checkpoint = "sic_diffusion_latest.pt"

    # Resumes training from checkpoint
    if os.path.exists(LATEST_CHECKPOINT):
        print(f"Found checkpoint! Resuming training from {LATEST_CHECKPOINT}...")
        checkpoint = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)
        
        # Extract underlying model if wrapped in DataParallel
        model_to_load = diffusion.module if isinstance(diffusion, nn.DataParallel) else diffusion
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
    
    print(f"Starting training on {DEVICE}...")

    # Profiling a few steps to avoid a massive trace file
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler_trace'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    prof.start()

    # Start at start_epoch instead of 1 if resuming training
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        
        is_parallel = isinstance(diffusion, PyG_DataParallel)

        # Training pass
        diffusion.train() # Sets modules like Dropout/BatchNorm into training mode
        total_train_loss = 0
        
        for batch_idx, data_list in enumerate(train_loader):
            optimizer.zero_grad() # Clears old gradients (otherwise they accumulate)
            
            # Routes through DataParallel if multiple GPUs are active
            if is_parallel:
                loss = diffusion(data_list)
            else:
                from torch_geometric.data import Batch
                data = Batch.from_data_list(data_list).to(DEVICE)
                loss = diffusion.get_loss(data)

            if loss.dim() > 0:
                loss = loss.mean() # Averages loss across GPUs
                
            loss.backward() # Computes gradients w.r.t. model params
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0) # strict speed limit on how much the model's weights can change in a single step
            optimizer.step() # Updates weights
            total_train_loss += loss.item() # converts scalar tensor to Python float for logging

            prof.step() # Step the profile forward

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation pass
        diffusion.eval() # Freeze layers like Dropout/BatchNorm (if any)
        total_val_loss = 0
        
        with torch.no_grad(): # Don't calculate gradients during validation (saves memory/time)
            for batch_idx, data_list in enumerate(val_loader):
                
                # Routes through DataParallel
                if is_parallel:
                    loss = diffusion(data_list)
                else:
                    from torch_geometric.data import Batch
                    data = Batch.from_data_list(data_list).to(DEVICE)
                    loss = diffusion.get_loss(data)

                if loss.dim() > 0:
                    loss = loss.mean()
                    
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Tracks training metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        if epoch % LOG_INTERVAL == 0:
            print(f"Epoch: {epoch:05d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Extracts model without DataParallel wrapper for saving
        model_to_save = diffusion.module if isinstance(diffusion, nn.DataParallel) else diffusion

        # Saves state to protect against Slurm timeouts
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }, LATEST_CHECKPOINT)

        # Best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model_to_save.state_dict(), BEST_CHECKPOINT)
            print(f"   -> New best validation loss! Overwriting {BEST_CHECKPOINT}")

    prof.stop() # End the profiler

    # Save final model
    model_to_save = diffusion.module if isinstance(diffusion, nn.DataParallel) else diffusion
    torch.save(model_to_save.state_dict(), FINAL_CHECKPOINT)
    print("Training Complete. Final model saved.")

    wandb.finish() # Closes the W&B run safely

    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Diffusion Model Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_CURVE_IMG)
    print(f"Saved loss curve to {LOSS_CURVE_IMG}")

if __name__ == "__main__":
    if not os.path.exists('processed'):
        os.makedirs('processed')
    train()