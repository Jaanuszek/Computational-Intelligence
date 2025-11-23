"""
Check training progress and checkpoint info
"""
import torch

checkpoint_path = 'best_ffdnet_checkpoint.pth'
model_path = 'ffdnet_model.pth'

print("Checking checkpoint information...\n")

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load best checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

print(f"{'='*60}")
print(f"BEST CHECKPOINT INFO ({checkpoint_path})")
print(f"{'='*60}")
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Validation Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
print(f"{'='*60}\n")
