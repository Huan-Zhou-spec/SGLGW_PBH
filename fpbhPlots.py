#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:50:19 2025

@author: ubuntu
"""

import numpy as np
import matplotlib.pyplot as plt
from modules import read_hdf5_data, create_interpolators


#使用插值画出的图
def plot_analysis_results_i(h5_filename, save_plots=True):
    """
    Analyze and plot data from HDF5 file using interpolation
    """
    # Read data
    data = read_hdf5_data(h5_filename)
    a_values = data['a_values']
    f_pbh_values = data['f_pbh_values']
    models_data = data['models_data']
    
    # Create interpolators
    interpolators = create_interpolators(data)
    
    # Create dense grids for smooth plotting
    a_dense = np.logspace(np.log10(a_values[0]), np.log10(a_values[-1]), 200)
    f_pbh_dense = np.logspace(-4, -2, 200)
    

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors and styles for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', '-']
    
    # Subplot 1: Compare first and last f_pbh, integral results vs a (all models)
    ax1 = axes[0]
    
    # 选取f_pbh_dense首尾两个 f_pbh 值
    first_fpbh = f_pbh_dense[0]
    last_fpbh = f_pbh_dense[-1]
    
    print(f"第一个 f_pbh 值: {first_fpbh:.2e}")
    print(f"最后一个 f_pbh 值: {last_fpbh:.2e}")
    
    for idx, (model_name, model_data) in enumerate(models_data.items()):
        color = colors[idx % len(colors)]
        interp_dict = interpolators[model_name]
        
        cosmo_type = model_data['cosmology_type']
        mode = model_data['mode']
        
        exponent_last = round(np.log10(last_fpbh))
        exponent_first = round(np.log10(first_fpbh))
        
        # 根据mode的值进行替换
        if cosmo_type == 'CDM' and mode == 'pos':
            mode_display = ''
            label_last = rf'{cosmo_type} {mode_display}'
            label_first = rf'{cosmo_type} {mode_display}'
        elif cosmo_type == 'CDM+PBH' and mode == 'pos':
            mode_display = 'Poisson'
            label_last = rf'{cosmo_type} {mode_display} ($f_{{\rm PBH}}$=$10^{{{exponent_last}}}$)'
            label_first = rf'{cosmo_type} {mode_display} ($f_{{\rm PBH}}$=$10^{{{exponent_first}}}$)'
        elif cosmo_type == 'CDM+PBH' and mode == 'clu':
            mode_display = 'Cluster'
            label_last = rf'{cosmo_type} {mode_display} ($f_{{\rm PBH}}$=$10^{{{exponent_last}}}$)'
            label_first = rf'{cosmo_type} {mode_display} ($f_{{\rm PBH}}$=$10^{{{exponent_first}}}$)'
        else:
            mode_display = mode
        
        # 为最后一个 f_pbh 值画图（使用密集网格）
        points_last = np.column_stack([np.full_like(a_dense, last_fpbh), a_dense])
        integral_dense_last = interp_dict['integral'](points_last)
        ax1.semilogx(a_dense, a_dense * integral_dense_last, 
                  color=color, linestyle='-', alpha=0.7, 
                  label=label_last, linewidth=2)
        
        if cosmo_type == 'CDM':
            continue
        
        # 为第一个 f_pbh 值画图（使用密集网格）
        points_first = np.column_stack([np.full_like(a_dense, first_fpbh), a_dense])
        integral_dense_first = interp_dict['integral'](points_first)
        ax1.semilogx(a_dense, a_dense * integral_dense_first, 
                  color=color, linestyle='--', label=label_first, linewidth=2)
    
    ax1.set_xlabel(r'$\Delta t$ (hrs)', fontsize=18)
    ax1.set_ylabel(r'$\Delta t\times p(\Delta t|\Omega)$', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_xlim(a_values[0], a_values[-1])
    ax1.legend(fontsize=15)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: N_lens vs f_pbh for different models (使用插值)
    ax2 = axes[1]
    
    for idx, (model_name, model_data) in enumerate(models_data.items()):
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        interp_dict = interpolators[model_name]
        
        cosmo_type = model_data['cosmology_type']
        mode = model_data['mode']
        
        # 根据mode的值进行替换
        if cosmo_type == 'CDM' and mode == 'pos':
            mode_display = ''
        elif cosmo_type == 'CDM+PBH' and mode == 'pos':
            mode_display = 'Poisson'
        elif cosmo_type == 'CDM+PBH' and mode == 'clu':
            mode_display = 'Cluster'
        else:
            mode_display = mode
        
        # 绘制N_lens vs f_pbh
        label = f'{cosmo_type} {mode_display}'
        N_lens_dense = interp_dict['N_lens'](f_pbh_dense)
        ax2.semilogx(f_pbh_dense, N_lens_dense, 
                  color=color, linestyle=linestyle, label=label, linewidth=2)
    
    ax2.set_xlabel(r'$f_{\rm PBH}$', fontsize=18)
    ax2.set_ylabel(r'$\Lambda_{\rm L,GW}(\Omega)$', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.set_xlim(f_pbh_dense[0], f_pbh_dense[-1])
    ax2.legend(fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    # 打印插值后的N_lens值用于验证
    print("\n插值后的 N_lens 值范围:")
    for idx, (model_name, model_data) in enumerate(models_data.items()):
        interp_dict = interpolators[model_name]
        N_lens_min = interp_dict['N_lens'](f_pbh_values[0])
        N_lens_max = interp_dict['N_lens'](f_pbh_values[-1])
        print(f"{model_name} - N_lens range: {N_lens_min:.2e} to {N_lens_max:.2e}")
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('Plots/analysis_results_interpolated.pdf', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return data



#使用原始数据画出的图
def plot_analysis_results_o(h5_filename, save_plots=True):
    """
    Analyze and plot data from HDF5 file
    """
    # Read data
    data = read_hdf5_data(h5_filename)
    a_values = data['a_values']
    f_pbh_values = data['f_pbh_values']
    models_data = data['models_data']
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors and styles for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', '-']
    
    # Subplot 1: Compare first and last f_pbh, integral results vs a (all models)
    ax1 = axes[0]
    
    # 选取首尾两个 f_pbh 索引
    first_fpbh_idx = 0
    last_fpbh_idx = len(f_pbh_values) - 1
    
    print(f"第一个 f_pbh 值: {f_pbh_values[first_fpbh_idx]:.2e}")
    print(f"最后一个 f_pbh 值: {f_pbh_values[last_fpbh_idx]:.2e}")
    
    for idx, (model_name, model_data) in enumerate(models_data.items()):
        color = colors[idx % len(colors)]
        
        integral_results = model_data['integral_results']
        cosmo_type = model_data['cosmology_type']
        mode = model_data['mode']
        
        
        exponent_last = round(np.log10(f_pbh_values[last_fpbh_idx]))
        exponent_first = round(np.log10(f_pbh_values[first_fpbh_idx]))
        # 根据mode的值进行替换
        if cosmo_type == 'CDM' and mode == 'pos':
            mode_display = ''
            label_last = rf'{cosmo_type} {mode_display}'
            label_first = rf'{cosmo_type} {mode_display}'
        elif cosmo_type == 'CDM+PBH' and mode == 'pos':
            mode_display = 'Poisson'
            label_last = rf'{cosmo_type} {mode_display} ($f_{{\rm PBH}}$=$10^{{{exponent_last}}}$)'
            label_first = rf'{cosmo_type} {mode_display} ($f_{{\rm PBH}}$=$10^{{{exponent_first}}}$)'
        elif cosmo_type == 'CDM+PBH' and mode == 'clu':
            mode_display = 'Cluster'
            label_last = rf'{cosmo_type} {mode_display} ($f_{{\rm PBH}}$=$10^{{{exponent_last}}}$)'
            label_first = rf'{cosmo_type} {mode_display} ($f_{{\rm PBH}}$=$10^{{{exponent_first}}}$)'
        else:
            mode_display = mode
        
        
        # 为最后一个 f_pbh 值画图（虚线）
        ax1.semilogx(a_values, a_values*integral_results[last_fpbh_idx, :], 
                  color=color, linestyle='-', alpha=0.7, 
                  label=label_last, linewidth=2)
        
        if cosmo_type == 'CDM':
            continue
        
        # 为第一个 f_pbh 值画图（实线）
        ax1.semilogx(a_values, a_values*integral_results[first_fpbh_idx, :], 
                  color=color, linestyle='--', label=label_first, linewidth=2)
        
        
    
    ax1.set_xlabel(r'$\Delta t$ (hrs)', fontsize=18)
    ax1.set_ylabel(r'$\Delta t\times p(\Delta t|\Omega)$', fontsize=18)
    ax1.tick_params(axis='both', which='major', 
                  labelsize=18)
    ax1.set_xlim(a_values[0], a_values[-1])
    ax1.legend(fontsize=15)  # 减小字体以适应更多图例项
    ax1.grid(True, alpha=0.3)
    
    
    # Subplot 2: N_lens vs f_pbh for different models
    ax2 = axes[1]
    
    for idx, (model_name, model_data) in enumerate(models_data.items()):
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        
        # 获取N_lens数据
        N_lens = model_data['N_lens']
        cosmo_type = model_data['cosmology_type']
        mode = model_data['mode']
        
        # 根据mode的值进行替换
        if cosmo_type == 'CDM' and mode == 'pos':
            mode_display = ''
        elif cosmo_type == 'CDM+PBH' and mode == 'pos':
            mode_display = 'Poisson'
        elif cosmo_type == 'CDM+PBH' and mode == 'clu':
            mode_display = 'Cluster'
        else:
            mode_display = mode
        
        # 绘制N_lens vs f_pbh
        label = f'{cosmo_type} {mode_display}'
        ax2.semilogx(f_pbh_values, N_lens, 
                  color=color, linestyle=linestyle, label=label, linewidth=2)
    
    f_pbh_dense = np.logspace(-4, -2, 200)
    ax2.set_xlabel(r'$f_{\rm PBH}$', fontsize=18)
    ax2.set_ylabel(r'$\Lambda_{\rm L,GW}(\Omega)$', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.set_xlim(f_pbh_dense[0], f_pbh_dense[-1])
    ax2.legend(fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    # 打印N_lens值用于验证
    for idx, (model_name, model_data) in enumerate(models_data.items()):
        N_lens = model_data['N_lens']
        print(f"{model_name} - N_lens range: {N_lens.min():.2e} to {N_lens.max():.2e}")
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('Plots/analysis_results_original.pdf', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return data


# Example usage
if __name__ == "__main__":
    h5_filename = 'data/lensing_analysis_data/lensing_analysis_results.h5'
    data = plot_analysis_results_i(h5_filename)