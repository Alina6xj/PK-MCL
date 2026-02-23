import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.generate_schrodinger import build_dataset
import torch
import numpy as np

print("Testing Schrodinger data generation...")
ds = build_dataset(n_traj=1, n_train=1, t_end=0.5, nx=512)
torch.save(ds, 'data/schrodinger_test.pt')
print('Dataset saved!')
print('train_real shape:', ds['train_real'].shape)
print('train_imag shape:', ds['train_imag'].shape)
print('test_real shape:', ds['test_real'].shape)
print('test_imag shape:', ds['test_imag'].shape)
print('meta:', ds['meta'])

real_data = ds['train_real'][0].numpy()
imag_data = ds['train_imag'][0].numpy()
print('\n--- Statistics ---')
print(f'real: min={np.min(real_data):.6f}, max={np.max(real_data):.6f}, mean={np.mean(real_data):.6f}')
print(f'imag: min={np.min(imag_data):.6f}, max={np.max(imag_data):.6f}, mean={np.mean(imag_data):.6f}')
print(f'real has NaN: {np.isnan(real_data).any()}')
print(f'imag has NaN: {np.isnan(imag_data).any()}')
print(f'real has Inf: {np.isinf(real_data).any()}')
print(f'imag has Inf: {np.isinf(imag_data).any()}')
