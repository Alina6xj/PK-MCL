import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.generate_ns import build_dataset
import torch
import numpy as np

print("Testing NS data generation...")
ds = build_dataset(n_traj=1, n_train=1, t_end=0.1, nx=64, ny=64)
torch.save(ds, 'data/ns_test.pt')
print('Dataset saved!')
print('train_omega shape:', ds['train_omega'].shape)
print('test_omega shape:', ds['test_omega'].shape)
print('meta:', ds['meta'])

omega_data = ds['train_omega'][0].numpy()
print('\n--- Statistics ---')
print(f'omega: min={np.min(omega_data):.6f}, max={np.max(omega_data):.6f}, mean={np.mean(omega_data):.6f}')
print(f'omega has NaN: {np.isnan(omega_data).any()}')
print(f'omega has Inf: {np.isinf(omega_data).any()}')
