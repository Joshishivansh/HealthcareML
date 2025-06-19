import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from src.models.stego_gan import stegoganmodel
from src.dataloader import get_dataloaders
from src.util.visualizer import Visualizer
from src.util.util import mkdirs

# ---------- Config ----------
data_root = 'data'                     # root containing patient folders
split_dir = 'data/splits'             # split .txt files
save_dir = 'outputs/checkpoints'      # where models are saved
log_dir = 'outputs/logs'              # tensorboard logs
use_mask = False                      # use datasetmask.py or not
gpu_ids = [0]                         # change if using CPU or multiple GPUs
batch_size = 4
num_workers = 2
num_epochs = 100
save_freq = 10                        # epochs between saves
display_freq = 100                    # steps between visual logs
lr = 0.0002

# ---------- Setup ----------
device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
mkdirs([save_dir, log_dir])

# Dataloaders
train_loader, val_loader, _ = get_dataloaders(data_root, split_dir, batch_size, num_workers, use_mask)

# Model
model = stegoganmodel(input_nc=3, output_nc=3, ngf=64, ndf=64, gpu_ids=gpu_ids)
model.setup()  # includes scheduler setup and weight init
model = model.to(device)

# Logging
writer = SummaryWriter(log_dir)
visualizer = Visualizer(log_dir)
total_steps = 0

# ---------- Training ----------
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for i, data in enumerate(train_loader):
        total_steps += 1
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % display_freq == 0:
            visuals = model.get_current_visuals()
            visualizer.display_current_results(visuals, epoch, total_steps)

        losses = model.get_current_losses()
        epoch_loss += sum(losses.values())
        for k, v in losses.items():
            writer.add_scalar(f'train/{k}', v, total_steps)

    print(f"[Epoch {epoch}] Avg Loss: {epoch_loss / len(train_loader):.4f}")

    # Update learning rate
    model.update_learning_rate()

    # Save checkpoint
    if epoch % save_freq == 0:
        print(f"=> Saving model at epoch {epoch}")
        model.save_networks(epoch)

    # ---------- Validation ----------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            model.set_input(val_data)
            model.forward_only()
            val_losses = model.get_current_losses()
            val_loss += sum(val_losses.values())
    val_loss /= len(val_loader)
    print(f"[Epoch {epoch}] Validation Loss: {val_loss:.4f}")
    writer.add_scalar('val/total_loss', val_loss, epoch)

print("Training finished...................................................................")
writer.close()
