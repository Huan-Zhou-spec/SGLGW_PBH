#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 21:55:14 2026

@author: ubuntu
"""

import numpy as np
import matplotlib.pyplot as plt

def format_R_value(R):
    """格式化 R 值为科学计数法表示"""
    if R == 0:
        return "0"
    
    exponent = int(np.floor(np.log10(abs(R))))
    coefficient = R / 10**exponent
    
    if coefficient == 1:
        return f"$10^{{{exponent}}}$"
    elif coefficient == -1:
        return f"$-10^{{{exponent}}}$"
    else:
        return f"${coefficient:.1f} \\times 10^{{{exponent}}}$"

def plot_sampling_analysis_R_from_file(data_filename):
    """
    从已保存的数据文件加载数据并绘制R倍数的采样分析图形
    保持与原始run_sampling_analysis_R函数相同的绘图格式
    
    Parameters:
    -----------
    data_filename : str
        保存的数据文件路径
    """
    # 加载数据
    data = np.load(data_filename, allow_pickle=True)
    
    # 提取基本信息
    multipliers = data['multipliers']
    model_name = str(data['model_name'])
    N_lens = data['N_lens']
    R = data['R']
    lgT_obs = data['lgT_obs']
    lg_dt_new = data['lg_dt_grid']
    pdf_normalized = data['theoretical_pdf']
    cdf_theoretical = data['theoretical_cdf']
    
    # 计算T_obs的年单位值
    T_obs = 10**lgT_obs/365/24
    
    print(f"模型: {model_name}")
    print(f"基础R: {R:.2e} yr⁻¹")
    print(f"基础N_lens: {N_lens}")
    print(f"T_obs: {T_obs:.1f} 年")
    
    # 绘制统计图
    # 创建比较图形 - 包含四个子图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子图1: 多个倍数的PDF比较
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, multiplier in enumerate(multipliers):
        if i >= len(colors):
            color = colors[i % len(colors)]
        else:
            color = colors[i]
            
        # 构建键名后缀
        if multiplier == int(multiplier):
            multiplier_str = str(int(multiplier))
        else:
            multiplier_str = str(multiplier).replace('.', '_')
        
        # 提取样本数据
        samples_lg_dt = data[f'samples_lg_dt_{multiplier_str}']
        n_samples = int(data[f'n_samples_{multiplier_str}'])
        R_samples = data[f'R_samples_{multiplier_str}']
        
        # 绘制直方图
        R_formatted = format_R_value(R_samples)
        axes[0].hist(samples_lg_dt, bins=20, density=True, alpha=0.5, 
                       color=color, label=rf'R = {R_formatted} $\mathrm{{yr}}^{{-1}}$, n={n_samples}', 
                       histtype='step', linewidth=2)
    
    # 标记T_obs的位置
    axes[0].axvline(x=lgT_obs, color='k', linestyle='--', alpha=1,
                      label=rf'${{T_{{\rm obs}}}}$={T_obs:.1f} yrs')
    
    # 添加理论PDF
    axes[0].plot(lg_dt_new, pdf_normalized, 'k-', linewidth=3, label='Theoretical PDF', alpha=0.8)
    axes[0].set_xlabel(r'log$[\Delta t~(\rm hrs)]$', fontsize = 18)
    axes[0].set_ylabel('Probability Density', fontsize = 18)
    axes[0].set_xlim(lg_dt_new[0], lg_dt_new[-1]-2)
    axes[0].legend(fontsize=13)
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: 多个倍数的CDF比较
    for i, multiplier in enumerate(multipliers):
        if i >= len(colors):
            color = colors[i % len(colors)]
        else:
            color = colors[i]
            
        # 构建键名后缀
        if multiplier == int(multiplier):
            multiplier_str = str(int(multiplier))
        else:
            multiplier_str = str(multiplier).replace('.', '_')
        
        # 提取样本数据
        ks_statistic = data[f'ks_statistic_{multiplier_str}']
        R_samples = data[f'R_samples_{multiplier_str}']
        sample_cdf_sorted = data[f'sample_cdf_sorted_{multiplier_str}']
        sample_cdf_values = data[f'sample_cdf_values_{multiplier_str}']
        
        # 在标签中使用
        R_formatted = format_R_value(R_samples)
        axes[1].plot(sample_cdf_sorted, sample_cdf_values, 
                       color=color, linewidth=2, 
                       label=rf'R = {R_formatted} $\mathrm{{yr}}^{{-1}}$, KS={ks_statistic:.3f}')
        
    # 标记T_obs的位置
    axes[1].axvline(x=lgT_obs, color='k', linestyle='--', alpha=1,
                      label=rf'${{T_{{\rm obs}}}}$={T_obs:.1f} yrs')
    # 添加理论CDF
    axes[1].plot(lg_dt_new, cdf_theoretical, 'k-', linewidth=1.5, label='Theoretical CDF', alpha=0.8)
    axes[1].set_xlabel(r'log$[\Delta t~(\rm hrs)]$', fontsize = 18)
    axes[1].set_ylabel('Cumulative Probability', fontsize = 18)
    axes[1].set_xlim(lg_dt_new[0], lg_dt_new[-1]-2)
    axes[1].legend(fontsize=13)
    axes[1].tick_params(axis='both', which='major', labelsize=18)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_filename = 'Plots/time_delay_sampling_R_comparison.pdf'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计摘要
    print("\n" + "="*50)
    print("Sampling Statistics Summary (Loaded from file)")
    print("="*50)
    for multiplier in multipliers:
        # 构建键名后缀
        if multiplier == int(multiplier):
            multiplier_str = str(int(multiplier))
        else:
            multiplier_str = str(multiplier).replace('.', '_')
        
        ks_stat = data[f'ks_statistic_{multiplier_str}']
        n_samples = int(data[f'n_samples_{multiplier_str}'])
        R_samples = data[f'R_samples_{multiplier_str}']
        R_formatted = format_R_value(R_samples)
        print(f"Multiplier {multiplier}: R = {R_formatted} yr⁻¹, {n_samples} samples, KS statistic = {ks_stat:.6f}")
    
    return output_filename


def plot_sampling_analysis_T_from_file(data_filename):
    """
    从已保存的数据文件加载数据并绘制T倍数的采样分析图形
    保持与原始run_sampling_analysis_T函数相同的绘图格式
    
    Parameters:
    -----------
    data_filename : str
        保存的数据文件路径
    """
    # 加载数据
    data = np.load(data_filename, allow_pickle=True)
    
    # 提取基本信息
    multipliers_T = data['multipliers_T']
    model_name = str(data['model_name'])
    multiplier_R = data['multiplier_R']
    
    print(f"模型: {model_name}")
    print(f"R倍数: {multiplier_R}")
    print(f"T倍数: {multipliers_T}")
    
    # 绘制统计图
    # 创建比较图形 - 包含两个子图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子图1: 多个观测时间倍数的PDF比较
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, multiplier_T in enumerate(multipliers_T):
        if i >= len(colors):
            color = colors[i % len(colors)]
        else:
            color = colors[i]
            
        # 构建键名后缀
        if multiplier_T == int(multiplier_T):
            multiplier_str = str(int(multiplier_T))
        else:
            multiplier_str = str(multiplier_T).replace('.', '_')
        
        # 提取数据
        samples_lg_dt = data[f'samples_lg_dt_T_{multiplier_str}']
        n_samples = int(data[f'n_samples_T_{multiplier_str}'])
        lgT_obs = data[f'lgT_obs_T_{multiplier_str}']
        lg_dt_grid = data[f'lg_dt_grid_T_{multiplier_str}']
        pdf_normalized = data[f'pdf_normalized_T_{multiplier_str}']
        
        # 绘制直方图
        T_obs_years = 10**lgT_obs/365/24
        axes[0].hist(samples_lg_dt, bins=20, density=True, alpha=0.5, 
                       color=color, label=rf'$T_{{\rm obs}}$={T_obs_years:.1f} yrs, n={n_samples}', 
                       histtype='step', linewidth=2)
        
        # 添加理论PDF
        axes[0].plot(lg_dt_grid, pdf_normalized, color=color, linewidth=3, 
                     alpha=0.8)

        # 标记T_obs的位置
        axes[0].axvline(x=lgT_obs, color='k', linestyle='--', alpha=1)
    
    axes[0].set_xlabel(r'log$[\Delta t~(\rm hrs)]$', fontsize = 18)
    axes[0].set_ylabel('Probability Density', fontsize = 18)
    axes[0].set_xlim(lg_dt_grid[0], lg_dt_grid[-1]-2)
    axes[0].legend(fontsize=13)
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: 多个观测时间倍数的CDF比较
    for i, multiplier_T in enumerate(multipliers_T):
        if i >= len(colors):
            color = colors[i % len(colors)]
        else:
            color = colors[i]
            
        # 构建键名后缀
        if multiplier_T == int(multiplier_T):
            multiplier_str = str(int(multiplier_T))
        else:
            multiplier_str = str(multiplier_T).replace('.', '_')
        
        # 提取数据
        ks_statistic = data[f'ks_statistic_T_{multiplier_str}']
        lgT_obs = data[f'lgT_obs_T_{multiplier_str}']
        lg_dt_grid = data[f'lg_dt_grid_T_{multiplier_str}']
        cdf_theoretical = data[f'cdf_theoretical_T_{multiplier_str}']
        samples_lg_dt = data[f'samples_lg_dt_T_{multiplier_str}']
        
        # 计算样本的CDF（因为数据文件中没有保存CDF数据）
        sample_cdf_sorted = np.sort(samples_lg_dt)
        sample_cdf_values = np.arange(1, len(sample_cdf_sorted)+1) / len(sample_cdf_sorted)
        
        # 在标签中使用
        T_obs_years = 10**lgT_obs/365/24
        axes[1].plot(sample_cdf_sorted, sample_cdf_values, 
                       color=color, linewidth=3, 
                       label = rf'$T_{{\rm obs}}$={T_obs_years:.1f} yrs, KS={ks_statistic:.3f}')
        
        # 标记T_obs的位置
        axes[1].axvline(x=lgT_obs, color='k', linestyle='--', alpha=1)
        
        # 添加理论CDF
        axes[1].plot(lg_dt_grid, cdf_theoretical, 'k-', linewidth=1.5, alpha=0.8)
    
    axes[1].set_xlabel(r'log$[\Delta t~(\rm hrs)]$', fontsize = 18)
    axes[1].set_ylabel('Cumulative Probability', fontsize = 18)
    axes[1].set_xlim(lg_dt_grid[0], lg_dt_grid[-1]-2)
    axes[1].legend(fontsize=13)
    axes[1].tick_params(axis='both', which='major', labelsize=18)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_filename = 'Plots/time_delay_sampling_T_comparison.pdf'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计摘要
    print("\n" + "="*50)
    print("Sampling Statistics Summary (T multipliers, Loaded from file)")
    print("="*50)
    for multiplier_T in multipliers_T:
        # 构建键名后缀
        if multiplier_T == int(multiplier_T):
            multiplier_str = str(int(multiplier_T))
        else:
            multiplier_str = str(multiplier_T).replace('.', '_')
        
        ks_stat = data[f'ks_statistic_T_{multiplier_str}']
        n_samples = int(data[f'n_samples_T_{multiplier_str}'])
        lgT_obs = data[f'lgT_obs_T_{multiplier_str}']
        R = data[f'R_T_{multiplier_str}']
        T_obs_years = 10**lgT_obs/365/24
        print(f"Multiplier_T {multiplier_T}: T_obs={T_obs_years:.2f} yrs, R={R:.2e} yr⁻¹, {n_samples} samples, KS statistic = {ks_stat:.6f}")
    
    return output_filename


# 绘制R倍数比较图（使用已有的npz文件）
plot_sampling_analysis_R_from_file('data/simulation_data/CDM_data/time_delay_samples_CDM_all_multipliers_R.npz')

# 绘制T倍数比较图（使用已有的npz文件）
plot_sampling_analysis_T_from_file('data/simulation_data/CDM_data/time_delay_samples_CDM_all_multipliers_T.npz')