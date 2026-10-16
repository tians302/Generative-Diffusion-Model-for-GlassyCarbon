import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from dataset import GCGraphDataset
from nequip_layer import NequIP_GC
from diffusion import DiffusionModel
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel as PyG_DataParallel
from torch_geometric.data import Batch
import os
from config_loader import load_config
from train_logger import TrainLogger
# import wandb
# import torch.profiler

config = load_config()

# TRAINING HYPERPARAMETERS
BATCH_SIZE = config['training']['batch_size']
LEARNING_RATE = float(config['training']['learning_rate'])
NUM_EPOCHS = config['training']['num_epochs']
VAL_SPLIT = config['training']['val_split']
LOG_INTERVAL = config['training']['log_interval']
CHECKPOINT_INTERVAL = config['training'].get('checkpoint_interval', LOG_INTERVAL)
USE_DATA_PARALLEL = config['training'].get('use_data_parallel', False)
NUM_WORKERS = config['training'].get('num_workers', 0)
SPLIT_SEED = config['training'].get('split_seed', 42)
WEIGHT_LOSS_BY_SIGMA = config['training'].get('weight_loss_by_sigma', True)
VAL_FIXED_TIMESTEPS = config['training'].get('val_fixed_timesteps', True)
VAL_T_PROBES = config['training'].get('val_t_probes', 8)
PLOT_SMOOTH_WINDOW = config['training'].get('plot_smooth_window', 50)
LOSS_RECIPE = "sigma_weighted_v1" if WEIGHT_LOSS_BY_SIGMA else "unweighted_v0"

PROCESSED_XYZ = config['data']['processed_xyz']
LATEST_CHECKPOINT = config['training']['checkpoints']['latest']
BEST_CHECKPOINT = config['training']['checkpoints']['best']
FINAL_CHECKPOINT = config['training']['checkpoints']['final']
MONITOR_IMG = config['training'].get('monitor_img', 'training_monitor.png')
METRICS_CSV = config['training']['metrics_csv']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def train():
    # wandb.init(
    #     project="gc-diffusion-benchmark",
    #     config=config,
    #     name="multi-gpu-benchmark"
    # )

    if not os.path.exists(PROCESSED_XYZ):
        print(f"Dataset '{PROCESSED_XYZ}' not found. Running data_ingestion.py...")
        import data_ingestion

    # Loads and splits dataset
    dataset = GCGraphDataset(root='.', xyz_file=PROCESSED_XYZ)
    dataset_size = len(dataset)
    val_size = max(1, int(dataset_size * VAL_SPLIT))
    train_size = dataset_size - val_size
    split_gen = torch.Generator().manual_seed(SPLIT_SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_gen)
    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == 'cuda')
    if NUM_WORKERS > 0:
        loader_kwargs['persistent_workers'] = True
    train_loader = DataListLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataListLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Initializing models
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
    
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    n_gpus = torch.cuda.device_count()
    if USE_DATA_PARALLEL and n_gpus > 1 and BATCH_SIZE >= n_gpus:
        print(f"Using {n_gpus} GPUs via PyG DataParallel.")
        diffusion = PyG_DataParallel(diffusion)
    elif n_gpus > 1:
        print(f"Found {n_gpus} GPUs; using 1 GPU (set training.use_data_parallel: true to shard).")

    diffusion.to(DEVICE)
    
    optimizer = optim.Adam(diffusion.parameters(), lr=LEARNING_RATE)
    
    # Tracking Variables & Resume Logic
    start_epoch = 1
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    if os.path.exists(LATEST_CHECKPOINT):
        print(f"Found checkpoint! Resuming training from {LATEST_CHECKPOINT}...")
        checkpoint = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)
        unwrap_model(diffusion).load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        if checkpoint.get('loss_recipe') != LOSS_RECIPE:
            print(
                f"Loss recipe changed ({checkpoint.get('loss_recipe')} -> {LOSS_RECIPE}). "
                "Resetting best/val history so old unweighted scores are not compared."
            )
            best_val_loss = float('inf')
            train_losses = []
            val_losses = []
            if os.path.exists(METRICS_CSV):
                archived = "training_metrics_pre_weighted.csv"
                os.replace(METRICS_CSV, archived)
                print(f"Archived previous metrics to {archived}")

    # Skip t=0. sigma is 0.001 there, and dividing the error by it blows up validation.
    probe = torch.linspace(1, config['diffusion']['num_steps'] - 1, VAL_T_PROBES).long()
    val_t_all = probe.repeat((val_size + VAL_T_PROBES - 1) // VAL_T_PROBES)[:val_size]
    
    logger = TrainLogger(
        csv_path=METRICS_CSV,
        plot_path=MONITOR_IMG,
        plot_interval=LOG_INTERVAL,
        resume=start_epoch > 1 and os.path.exists(METRICS_CSV),
        smooth_window=PLOT_SMOOTH_WINDOW,
    )
    print(
        f"Starting training on {DEVICE} | batch_size={BATCH_SIZE} | "
        f"{len(train_loader)} train batches/epoch | recipe={LOSS_RECIPE} | "
        f"fixed_val_t={VAL_FIXED_TIMESTEPS} | logging to {METRICS_CSV}"
    )

    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler_trace'),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # )
    # prof.start()

    # Start at start_epoch instead of 1 if resuming training
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        
        is_parallel = isinstance(diffusion, PyG_DataParallel)

        # Training pass
        diffusion.train()
        total_train_loss = 0.0
        
        for data_list in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            if is_parallel:
                loss = diffusion(data_list)
            else:
                data = Batch.from_data_list(data_list).to(DEVICE)
                loss = diffusion.get_loss(data, weight_by_sigma=WEIGHT_LOSS_BY_SIGMA)

            if loss.dim() > 0:
                loss = loss.mean()
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.detach()

        avg_train_loss = (total_train_loss / len(train_loader)).item()
        train_losses.append(avg_train_loss)

        # Validation pass
        diffusion.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            graph_offset = 0
            for data_list in val_loader:
                n_graphs = len(data_list)
                t_batch = val_t_all[graph_offset:graph_offset + n_graphs] if VAL_FIXED_TIMESTEPS else None
                graph_offset += n_graphs
                if is_parallel:
                    loss = diffusion(data_list)
                else:
                    data = Batch.from_data_list(data_list).to(DEVICE)
                    loss = diffusion.get_loss(
                        data,
                        t=t_batch,
                        weight_by_sigma=WEIGHT_LOSS_BY_SIGMA,
                    )

                if loss.dim() > 0:
                    loss = loss.mean()
                    
                total_val_loss += loss.detach()
                
        avg_val_loss = (total_val_loss / len(val_loader)).item()
        val_losses.append(avg_val_loss)

        # wandb.log({
        #     "epoch": epoch,
        #     "train_loss": avg_train_loss,
        #     "val_loss": avg_val_loss
        # })

        model_to_save = unwrap_model(diffusion)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model_to_save.state_dict(), BEST_CHECKPOINT)
            print(f"   -> New best validation loss! Overwriting {BEST_CHECKPOINT}")

        logger.log(
            epoch=epoch,
            num_epochs=NUM_EPOCHS,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            best_val_loss=best_val_loss,
            lr=optimizer.param_groups[0]['lr'],
            train_losses=train_losses,
            val_losses=val_losses,
        )

        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == NUM_EPOCHS:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'loss_recipe': LOSS_RECIPE,
            }, LATEST_CHECKPOINT)

    # prof.stop()

    model_to_save = unwrap_model(diffusion)
    torch.save(model_to_save.state_dict(), FINAL_CHECKPOINT)
    print(f"Training complete. Metrics: {METRICS_CSV}  Monitor: {MONITOR_IMG}")

    # wandb.finish()

if __name__ == "__main__":
    if not os.path.exists('processed'):
        os.makedirs('processed')
    train()