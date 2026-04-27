import os
import torch

# ==========================================
# FIX FOR PYTORCH 2.6+ AND OGB INCOMPATIBILITY
# Force torch.load to use the old behavior 
# so OGB can unpickle its NumPy dictionaries.
# ==========================================
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# ==========================================

from ogb.lsc import MAG240MDataset
#import os

# Make sure to run this from lustre (like /lustre/orion/[projid]/scratch/[userid]/)

print(f"Downloading and preprocessing MAG240M...")
print("This will download a 168GB zip file and unpack it to ~400GB.")
print("This may take a while depending on network and disk speeds...")

# Initializing the class triggers the download and extraction automatically.
dataset = MAG240MDataset(root="dataset/")

print("\nSuccess! Dataset is downloaded, extracted, and preprocessed.")
print(f"Total papers: {dataset.num_papers:,}")
