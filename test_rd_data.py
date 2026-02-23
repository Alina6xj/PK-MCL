import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.generate_rd import build_dataset
import torch
import numpy as np

print("Testing RD data generation...")
ds = build_dataset(n_traj=1, n_train=1, t_end=50.0, nx=100, ny=100)
torch.save(ds, 'data/rd_test.pt')
print('Dataset saved!')
print('train_u shape:', ds['train_u'].shape)
print('train_v shape:', ds['train_v'].shape)
print('test_u shape:', ds['test_u'].shape)
print('test_v shape:', ds['test_v'].shape)
print('meta:', ds['meta'])

u_data = ds['train_u'][0].numpy()
v_data = ds['train_v'][0].numpy()
print('\n--- Statistics ---')
print(f'u: min={np.min(u_data):.6f}, max={np.max(u_data):.6f}, mean={np.mean(u_data):.6f}')
print(f'v: min={np.min(v_data):.6f}, max={np.max(v_data):.6f}, mean={np.mean(v_data):.6f}')
print(f'u has NaN: {np.isnan(u_data).any()}')
print(f'v has NaN: {np.isnan(v_data).any()}')
print(f'u has Inf: {np.isinf(u_data).any()}')
print(f'v has Inf: {np.isinf(v_data).any()}')
