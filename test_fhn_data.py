import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.generate_fhn import build_dataset
import torch
import numpy as np

print("Testing FHN data generation...")
ds = build_dataset(n_traj=1, n_train=1, t_end=0.001)
torch.save(ds, 'data/fhn_test.pt')
print('Dataset saved!')
print('train_v shape:', ds['train_v'].shape)
print('train_w shape:', ds['train_w'].shape)
print('test_v shape:', ds['test_v'].shape)
print('test_w shape:', ds['test_w'].shape)
print('meta:', ds['meta'])

v_data = ds['train_v'][0].numpy()
w_data = ds['train_w'][0].numpy()
print('\n--- Statistics ---')
print(f'v: min={np.min(v_data):.6f}, max={np.max(v_data):.6f}, mean={np.mean(v_data):.6f}')
print(f'w: min={np.min(w_data):.6f}, max={np.max(w_data):.6f}, mean={np.mean(w_data):.6f}')
print(f'v has NaN: {np.isnan(v_data).any()}')
print(f'w has NaN: {np.isnan(w_data).any()}')
print(f'v has Inf: {np.isinf(v_data).any()}')
print(f'w has Inf: {np.isinf(w_data).any()}')
