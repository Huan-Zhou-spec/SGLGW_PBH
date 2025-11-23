#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:43:28 2025

@author: ubuntu
"""

import numpy as np
from modules import compute_interpolated_data
import matplotlib.pyplot as plt
from scipy import stats

def generate_samples_from_distribution(lg_dt_new, pdf_normalized, n_samples=10000):
    """
    根据归一化的概率密度函数生成随机样本
    
    参数:
    - lg_dt_new: log10尺度的时间延迟数组
    - pdf_normalized: 归一化的概率密度函数数组 (pdt/dt_nom)
    - n_samples: 要生成的样本数量
    
    返回:
    - samples_lg_dt: log10尺度的样本
    - samples_dt: 线性尺度的样本
    """
    
    # 方法1: 使用累积分布函数和逆变换采样
    def inverse_transform_sampling(lg_dt, pdf, n_samples):
        # 计算累积分布函数 (CDF)
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]  # 归一化到[0,1]
        
        # 生成均匀分布的随机数
        u = np.random.uniform(0, 1, n_samples)
        
        # 使用线性插值找到对应的lg_dt值
        samples = np.interp(u, cdf, lg_dt)
        
        return samples

    #方法2：拒绝采样法
    # 使用逆变换采样 - 更高效
    samples_lg_dt = inverse_transform_sampling(lg_dt_new, pdf_normalized, n_samples)
    
    # 将log10尺度的样本转换为线性尺度
    samples_dt = 10 ** samples_lg_dt
    
    return samples_lg_dt, samples_dt


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


def run_sampling_analysis_R(h5_filename, model_name='model_1', multipliers_R=[1, 0.1]):
    """
    运行完整的采样分析流程，包括多个倍数的采样和比较
    
    Parameters:
    -----------
    h5_filename : str
        HDF5数据文件路径
    model_name : str
        要分析的模型名称
    multipliers : list
        采样倍数列表
    """
    # 使用默认的密集网格
    print("Using default dense grid:")
    multiplier_T = 1
    interpolated_data = compute_interpolated_data(h5_filename, multiplier_T)
    
    # 访问特定模型的数据
    model_data = interpolated_data['models'][model_name]
    R = interpolated_data['R']
    
    dt_new = interpolated_data['new_a_values']
    lg_dt_new = np.log10(dt_new)
    pdt = model_data['integrand_dense'][0,:]
    N_lens = round(model_data['N_lens_multiplied'][0])
    
    # 计算归一化常数
    dt_nom = np.trapezoid(pdt * np.log(10) * dt_new, lg_dt_new)
    
    # 归一化的概率密度函数
    pdf_normalized = pdt * np.log(10) * dt_new / dt_nom
    
    print(f"Normalization constant dt_nom: {dt_nom:.6e}")
    print(f"Integrated normalized PDF: {np.trapezoid(pdf_normalized, lg_dt_new):.6f}")
    
    # 计算理论累积分布
    cdf_theoretical = np.cumsum(pdf_normalized)
    cdf_theoretical = cdf_theoretical / cdf_theoretical[-1]  # 归一化
    
    # 生成数据
    # 存储所有样本数据
    all_samples_data = {}
    
    # 为每个倍数生成样本
    for multiplier in multipliers_R:
        n_samples = int(N_lens * multiplier)
        
        # 生成样本
        samples_lg_dt, samples_dt = generate_samples_from_distribution(
            lg_dt_new, pdf_normalized, n_samples
        )
        
        print(f"\nMultiplier {multiplier}:")
        print(f"Generated {n_samples} samples (N_lens × {multiplier})")
        print(f"Sample range in log10 scale: {samples_lg_dt.min():.2f} to {samples_lg_dt.max():.2f}")
        print(f"Sample range in linear scale: {samples_dt.min():.2e} to {samples_dt.max():.2e}")
        
        # 验证采样质量 - 计算KS统计量
        R_samples = R*multiplier
        lgT_obs = np.log10(interpolated_data['T_obs_hours'])
        
        # 计算样本的累积分布
        sample_cdf_sorted = np.sort(samples_lg_dt)
        sample_cdf_values = np.arange(1, len(sample_cdf_sorted)+1) / len(sample_cdf_sorted)
        
        # 计算理论累积分布在样本点上的值
        theory_cdf_values = np.interp(sample_cdf_sorted, lg_dt_new, cdf_theoretical)
        
        # KS统计量
        ks_statistic = np.max(np.abs(sample_cdf_values - theory_cdf_values))
        
        print(f"KS statistic: {ks_statistic:.4f}")
        
        # 保存样本数据
        all_samples_data[f'multiplier_{multiplier}'] = {
            'samples_lg_dt': samples_lg_dt,
            'samples_dt': samples_dt,
            'ks_statistic': ks_statistic,
            'n_samples': n_samples,
            'R_samples': R_samples,
            'sample_cdf_sorted': sample_cdf_sorted,
            'sample_cdf_values': sample_cdf_values
        }
    
    # 绘制统计图
    # 创建比较图形 - 包含四个子图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子图1: 多个倍数的PDF比较
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, multiplier in enumerate(multipliers_R):
        if i >= len(colors):
            color = colors[i % len(colors)]
        else:
            color = colors[i]
            
        data_key = f'multiplier_{multiplier}'
        samples_data = all_samples_data[data_key]
        n_samples = samples_data['n_samples']
        R_samples = samples_data['R_samples']
        
        # 绘制直方图
        R_formatted = format_R_value(R_samples)
        axes[0].hist(samples_data['samples_lg_dt'], bins=20, density=True, alpha=0.5, 
                       color=color, label=rf'R = {R_formatted} $\rm yr^{-1}$, n={n_samples}', 
                       histtype='step', linewidth=2)
    
    # 标记T_obs的位置
    T_obs = 10**lgT_obs/365/24
    axes[0].axvline(x=lgT_obs, color='k', linestyle='--', alpha=0.7,
                      label=rf'${{T_{{\rm obs}}}}$={T_obs:.1f} yrs')
    
    # 添加理论PDF
    axes[0].plot(lg_dt_new, pdf_normalized, 'k-', linewidth=3, label='Theoretical PDF', alpha=0.8)
    axes[0].set_xlabel(r'log($\Delta t$)', fontsize = 18)
    axes[0].set_ylabel('Probability Density', fontsize = 18)
    axes[0].set_xlim(lg_dt_new[0], lg_dt_new[-1]-2)
    axes[0].legend(fontsize=13)
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: 多个倍数的CDF比较
    for i, multiplier in enumerate(multipliers_R):
        if i >= len(colors):
            color = colors[i % len(colors)]
        else:
            color = colors[i]
            
        data_key = f'multiplier_{multiplier}'
        samples_data = all_samples_data[data_key]
        ks_statistic = samples_data['ks_statistic']
        R_samples = samples_data['R_samples']
        
        # 在标签中使用
        R_formatted = format_R_value(R_samples)
        axes[1].plot(samples_data['sample_cdf_sorted'], samples_data['sample_cdf_values'], 
                       color=color, linewidth=2, 
                       label = rf'R = {R_formatted} $\rm yr^{-1}$, KS={ks_statistic:.3f}')
                       
    
    # 添加理论CDF
    axes[1].plot(lg_dt_new, cdf_theoretical, 'k-', linewidth=1.5, label='Theoretical CDF', alpha=0.8)
    axes[1].set_xlabel(r'log($\Delta t$)', fontsize = 18)
    axes[1].set_ylabel('Cumulative Probability', fontsize = 18)
    axes[1].set_xlim(lg_dt_new[0], lg_dt_new[-1]-2)
    axes[1].legend(fontsize=13)
    axes[1].tick_params(axis='both', which='major', labelsize=18)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Plots/time_delay_sampling_R_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存数据
    # 将所有倍数的数据保存到一个文件中
    combined_output_filename = f'data/simulation_data/time_delay_samples_CDM_all_multipliers_R.npz'
    
    # 准备保存的数据
    save_data = {
        'multipliers': np.array(multipliers_R),
        'model_name': model_name,
        'N_lens': N_lens,
        'R': R,
        'lgT_obs': lgT_obs,
        'lg_dt_grid': lg_dt_new,
        'theoretical_pdf': pdf_normalized,
        'theoretical_cdf': cdf_theoretical
    }
    
    # 添加每个倍数的数据 - 修复键名格式问题
    for multiplier in multipliers_R:
        #利用以上生成的总数据
        data_key = f'multiplier_{multiplier}'
        samples_data = all_samples_data[data_key] 
        
        # 将浮点数转换为字符串，但保持原始格式，对于整数倍数（如1），使用"1"而不是"1.0"
        if multiplier == int(multiplier):
            multiplier_str = str(int(multiplier))
        else:
            multiplier_str = str(multiplier).replace('.', '_')
        
        save_data[f'samples_lg_dt_{multiplier_str}'] = samples_data['samples_lg_dt']
        save_data[f'samples_dt_{multiplier_str}'] = samples_data['samples_dt']
        save_data[f'ks_statistic_{multiplier_str}'] = samples_data['ks_statistic']
        save_data[f'n_samples_{multiplier_str}'] = samples_data['n_samples']
        save_data[f'R_samples_{multiplier_str}'] = samples_data['R_samples']
        save_data[f'sample_cdf_sorted_{multiplier_str}'] = samples_data['sample_cdf_sorted']
        save_data[f'sample_cdf_values_{multiplier_str}'] = samples_data['sample_cdf_values']
    
    # 保存所有数据到一个文件
    np.savez(combined_output_filename, **save_data)
    print(f"All multiplier data saved to: {combined_output_filename}")
    
    # 打印统计摘要
    print("\n" + "="*50)
    print("Sampling Statistics Summary")
    print("="*50)
    for multiplier in multipliers_R:
        data_key = f'multiplier_{multiplier}'
        ks_stat = all_samples_data[data_key]['ks_statistic']
        n_samples = all_samples_data[data_key]['n_samples']
        print(f"Multiplier {multiplier}: {n_samples} samples, KS statistic = {ks_stat:.6f}")
    
    return combined_output_filename


def run_sampling_analysis_T(h5_filename, model_name='model_1', multipliers_T=[1, 0.1]):
    """
    运行完整的采样分析流程，针对不同的观测时间倍数进行采样和比较
    
    Parameters:
    -----------
    h5_filename : str
        HDF5数据文件路径
    model_name : str
        要分析的模型名称
    multipliers_T : list
        观测时间倍数列表
    """
    # 固定R的倍数
    multiplier_R = 0.2
    
    # 存储所有样本数据
    all_samples_data = {}
    
    # 为每个观测时间倍数生成样本
    for multiplier_T in multipliers_T:
        print(f"\nProcessing multiplier_T {multiplier_T}:")
        
        # 使用指定的观测时间倍数
        interpolated_data = compute_interpolated_data(h5_filename, multiplier_T)
        
        # 访问特定模型的数据
        model_data = interpolated_data['models'][model_name]
        R = interpolated_data['R']*multiplier_R
        
        dt_new = interpolated_data['new_a_values']
        lg_dt_new = np.log10(dt_new)
        pdt = model_data['integrand_dense'][0,:]
        N_lens = round(model_data['N_lens_multiplied'][0]*multiplier_R)
        
        # 计算归一化常数
        dt_nom = np.trapezoid(pdt * np.log(10) * dt_new, lg_dt_new)
        
        # 归一化的概率密度函数
        pdf_normalized = pdt * np.log(10) * dt_new / dt_nom
        
        print(f"Normalization constant dt_nom: {dt_nom:.6e}")
        print(f"Integrated normalized PDF: {np.trapezoid(pdf_normalized, lg_dt_new):.6f}")
        
        # 计算理论累积分布
        cdf_theoretical = np.cumsum(pdf_normalized)
        cdf_theoretical = cdf_theoretical / cdf_theoretical[-1]  # 归一化
        
        # 生成样本 - 使用固定的R倍数
        n_samples = int(N_lens * multiplier_R)
        
        samples_lg_dt, samples_dt = generate_samples_from_distribution(
            lg_dt_new, pdf_normalized, n_samples
        )
        
        print(f"Generated {n_samples} samples (N_lens × {multiplier_R})")
        print(f"Sample range in log10 scale: {samples_lg_dt.min():.2f} to {samples_lg_dt.max():.2f}")
        print(f"Sample range in linear scale: {samples_dt.min():.2e} to {samples_dt.max():.2e}")
        
        # 验证采样质量 - 计算KS统计量
        lgT_obs = np.log10(interpolated_data['T_obs_hours'])
        
        # 计算样本的累积分布
        sample_cdf_sorted = np.sort(samples_lg_dt)
        sample_cdf_values = np.arange(1, len(sample_cdf_sorted)+1) / len(sample_cdf_sorted)
        
        # 计算理论累积分布在样本点上的值
        theory_cdf_values = np.interp(sample_cdf_sorted, lg_dt_new, cdf_theoretical)
        
        # KS统计量
        ks_statistic = np.max(np.abs(sample_cdf_values - theory_cdf_values))
        
        print(f"KS statistic: {ks_statistic:.4f}")
        
        # 保存样本数据
        all_samples_data[f'multiplier_T_{multiplier_T}'] = {
            'samples_lg_dt': samples_lg_dt,
            'samples_dt': samples_dt,
            'ks_statistic': ks_statistic,
            'n_samples': n_samples,
            'multiplier_T': multiplier_T,
            'lgT_obs': lgT_obs,
            'sample_cdf_sorted': sample_cdf_sorted,
            'sample_cdf_values': sample_cdf_values,
            'pdf_normalized': pdf_normalized,
            'cdf_theoretical': cdf_theoretical,
            'lg_dt_grid': lg_dt_new,
            'N_lens': N_lens,
            'R': R
        }
    
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
            
        data_key = f'multiplier_T_{multiplier_T}'
        samples_data = all_samples_data[data_key]
        n_samples = samples_data['n_samples']
        lgT_obs = samples_data['lgT_obs']
        lg_dt_grid = samples_data['lg_dt_grid']
        pdf_normalized = samples_data['pdf_normalized']
        
        # 绘制直方图
        T_obs_years = 10**lgT_obs/365/24
        axes[0].hist(samples_data['samples_lg_dt'], bins=20, density=True, alpha=0.5, 
                       color=color, label=rf'$T_{{\rm obs}}$={T_obs_years:.1f} yrs, n={n_samples}', 
                       histtype='step', linewidth=2)
        
        # 添加理论PDF
        axes[0].plot(lg_dt_grid, pdf_normalized, color=color, linewidth=3, 
                     alpha=0.8)

        # 标记T_obs的位置
        #T_obs_ref = 10**lgT_obs/365/24
        axes[0].axvline(x=lgT_obs, color='k', linestyle='--', alpha=0.7)
    
    axes[0].set_xlabel(r'log($\Delta t$)', fontsize = 18)
    axes[0].set_ylabel('Probability Density', fontsize = 18)
    axes[0].set_xlim(lg_dt_grid[0], lg_dt_grid[-1]-2)
    axes[0].legend(fontsize=15)
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: 多个观测时间倍数的CDF比较
    for i, multiplier_T in enumerate(multipliers_T):
        if i >= len(colors):
            color = colors[i % len(colors)]
        else:
            color = colors[i]
            
        data_key = f'multiplier_T_{multiplier_T}'
        samples_data = all_samples_data[data_key]
        ks_statistic = samples_data['ks_statistic']
        lgT_obs = samples_data['lgT_obs']
        lg_dt_grid = samples_data['lg_dt_grid']
        cdf_theoretical = samples_data['cdf_theoretical']
        
        # 在标签中使用
        T_obs_years = 10**lgT_obs/365/24
        axes[1].plot(samples_data['sample_cdf_sorted'], samples_data['sample_cdf_values'], 
                       color=color, linewidth=3, 
                       label = rf'$T_{{\rm obs}}$={T_obs_years:.1f} yrs, KS={ks_statistic:.3f}')
        
        # 添加理论CDF
        axes[1].plot(lg_dt_grid, cdf_theoretical, 'k-', linewidth=1.5, alpha=0.8)
    
    axes[1].set_xlabel(r'log($\Delta t$)', fontsize = 18)
    axes[1].set_ylabel('Cumulative Probability', fontsize = 18)
    axes[1].set_xlim(lg_dt_grid[0], lg_dt_grid[-1]-2)
    axes[1].legend(fontsize=15)
    axes[1].tick_params(axis='both', which='major', labelsize=18)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Plots/time_delay_sampling_T_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存数据
    # 将所有观测时间倍数的数据保存到一个文件中
    combined_output_filename = f'data/simulation_data/time_delay_samples_CDM_all_multipliers_T.npz'
    
    # 准备保存的数据
    save_data = {
        'multipliers_T': np.array(multipliers_T),
        'model_name': model_name,
        'multiplier_R': multiplier_R
    }
    
    # 添加每个观测时间倍数的数据
    for multiplier_T in multipliers_T:
        data_key = f'multiplier_T_{multiplier_T}'
        samples_data = all_samples_data[data_key]
        
        # 将浮点数转换为字符串，但保持原始格式
        if multiplier_T == int(multiplier_T):
            multiplier_str = str(int(multiplier_T))
        else:
            multiplier_str = str(multiplier_T).replace('.', '_')
        
        save_data[f'samples_lg_dt_T_{multiplier_str}'] = samples_data['samples_lg_dt']
        save_data[f'samples_dt_T_{multiplier_str}'] = samples_data['samples_dt']
        save_data[f'ks_statistic_T_{multiplier_str}'] = samples_data['ks_statistic']
        save_data[f'n_samples_T_{multiplier_str}'] = samples_data['n_samples']
        save_data[f'lgT_obs_T_{multiplier_str}'] = samples_data['lgT_obs']
        save_data[f'pdf_normalized_T_{multiplier_str}'] = samples_data['pdf_normalized']
        save_data[f'cdf_theoretical_T_{multiplier_str}'] = samples_data['cdf_theoretical']
        save_data[f'lg_dt_grid_T_{multiplier_str}'] = samples_data['lg_dt_grid']
        save_data[f'N_lens_T_{multiplier_str}'] = samples_data['N_lens']
        save_data[f'R_T_{multiplier_str}'] = samples_data['R']
    
    # 保存所有数据到一个文件
    np.savez(combined_output_filename, **save_data)
    print(f"All multiplier_T data saved to: {combined_output_filename}")
    
    # 打印统计摘要
    print("\n" + "="*50)
    print("Sampling Statistics Summary (T multipliers)")
    print("="*50)
    for multiplier_T in multipliers_T:
        data_key = f'multiplier_T_{multiplier_T}'
        ks_stat = all_samples_data[data_key]['ks_statistic']
        n_samples = all_samples_data[data_key]['n_samples']
        lgT_obs = all_samples_data[data_key]['lgT_obs']
        T_obs_years = 10**lgT_obs/365/24
        print(f"Multiplier_T {multiplier_T}: T_obs={T_obs_years:.1f} yrs, {n_samples} samples, KS statistic = {ks_stat:.6f}")
    
    return combined_output_filename


# 统一的加载函数
def load_sampling_data(filename, multiplier, multiplier_type='R'):
    """
    从组合文件中加载特定倍数的采样数据
    
    Parameters:
    -----------
    filename : str
        组合数据文件路径
    multiplier : float
        要加载的倍数
    multiplier_type : str
        倍数类型，'R' 或 'T'
    
    Returns:
    --------
    dict : 包含该倍数所有数据的字典
    """
    data = np.load(filename, allow_pickle=True)
    
    # 修复键名格式：与保存时保持一致
    if multiplier == int(multiplier):
        multiplier_str = str(int(multiplier))
    else:
        multiplier_str = str(multiplier).replace('.', '_')
    
    # 根据倍数类型确定键名前缀
    if multiplier_type.upper() == 'R':
        prefix = ""
        required_keys = [
            f'samples_lg_dt_{multiplier_str}',
            f'samples_dt_{multiplier_str}',
            f'ks_statistic_{multiplier_str}',
            f'n_samples_{multiplier_str}',
            f'R_samples_{multiplier_str}'
        ]
        # 检查键是否存在
        for key in required_keys:
            if key not in data:
                available_keys = [k for k in data.keys() if k.startswith('samples_lg_dt_')]
                raise KeyError(f"Key '{key}' not found in file. Available R multipliers: {available_keys}")
        
        result = {
            'samples_lg_dt': data[f'samples_lg_dt_{multiplier_str}'],
            'samples_dt': data[f'samples_dt_{multiplier_str}'],
            'ks_statistic': data[f'ks_statistic_{multiplier_str}'],
            'n_samples': data[f'n_samples_{multiplier_str}'],
            'R_samples': data[f'R_samples_{multiplier_str}'],
            'multipliers': data['multipliers'],
            'model_name': data['model_name'].item() if hasattr(data['model_name'], 'item') else data['model_name'],
            'N_lens': data['N_lens'],
            'R': data['R'],
            'lgT_obs': data['lgT_obs'],
            'lg_dt_grid': data['lg_dt_grid'],
            'theoretical_pdf': data['theoretical_pdf'],
            'theoretical_cdf': data['theoretical_cdf'],
            'multiplier_type': 'R'
        }
        
        # 可选：添加CDF数据（如果存在）
        if f'sample_cdf_sorted_{multiplier_str}' in data:
            result['sample_cdf_sorted'] = data[f'sample_cdf_sorted_{multiplier_str}']
            result['sample_cdf_values'] = data[f'sample_cdf_values_{multiplier_str}']
            
    elif multiplier_type.upper() == 'T':
        prefix = "T_"
        required_keys = [
            f'samples_lg_dt_T_{multiplier_str}',
            f'samples_dt_T_{multiplier_str}',
            f'ks_statistic_T_{multiplier_str}',
            f'n_samples_T_{multiplier_str}'
        ]
        
        # 检查键是否存在
        for key in required_keys:
            if key not in data:
                available_keys = [k for k in data.keys() if k.startswith('samples_lg_dt_T_')]
                raise KeyError(f"Key '{key}' not found in file. Available T multipliers: {available_keys}")
        
        result = {
            'samples_lg_dt': data[f'samples_lg_dt_T_{multiplier_str}'],
            'samples_dt': data[f'samples_dt_T_{multiplier_str}'],
            'ks_statistic': data[f'ks_statistic_T_{multiplier_str}'],
            'n_samples': data[f'n_samples_T_{multiplier_str}'],
            'lgT_obs': data[f'lgT_obs_T_{multiplier_str}'],
            'multipliers_T': data['multipliers_T'],
            'model_name': data['model_name'].item() if hasattr(data['model_name'], 'item') else data['model_name'],
            'multiplier_R': data['multiplier_R'],
            'pdf_normalized': data[f'pdf_normalized_T_{multiplier_str}'],
            'cdf_theoretical': data[f'cdf_theoretical_T_{multiplier_str}'],
            'lg_dt_grid': data[f'lg_dt_grid_T_{multiplier_str}'],
            'N_lens': data[f'N_lens_T_{multiplier_str}'],
            'R': data[f'R_T_{multiplier_str}'],
            'multiplier_type': 'T'
        }
        
        # 可选：添加CDF数据（如果存在）
        if f'sample_cdf_sorted_T_{multiplier_str}' in data:
            result['sample_cdf_sorted'] = data[f'sample_cdf_sorted_T_{multiplier_str}']
            result['sample_cdf_values'] = data[f'sample_cdf_values_T_{multiplier_str}']
    else:
        raise ValueError("multiplier_type must be 'R' or 'T'")
    
    return result


# 统一的获取可用倍数函数
def get_available_multipliers(filename, multiplier_type='R'):
    """
    获取组合文件中包含的所有倍数
    
    Parameters:
    -----------
    filename : str
        组合数据文件路径
    multiplier_type : str
        倍数类型，'R' 或 'T'
    
    Returns:
    --------
    list : 可用的倍数列表
    """
    data = np.load(filename, allow_pickle=True)
    
    # 根据倍数类型确定键名前缀
    if multiplier_type.upper() == 'R':
        prefix = "samples_lg_dt_"
    elif multiplier_type.upper() == 'T':
        prefix = "samples_lg_dt_T_"
    else:
        raise ValueError("multiplier_type must be 'R' or 'T'")
    
    # 从数据键中提取倍数
    multipliers = []
    for key in data.keys():
        if key.startswith(prefix):
            # 提取倍数部分
            multiplier_str = key.replace(prefix, '')
            # 将下划线转换回小数点
            if '_' in multiplier_str:
                multiplier = float(multiplier_str.replace('_', '.'))
            else:
                multiplier = int(multiplier_str)
            multipliers.append(multiplier)
    
    return sorted(multipliers)


# 统一的获取文件信息函数
def get_sampling_data_info(filename, multiplier_type='R'):
    """
    获取组合文件中所有数据的详细信息
    
    Parameters:
    -----------
    filename : str
        组合数据文件路径
    multiplier_type : str
        倍数类型，'R' 或 'T'
    
    Returns:
    --------
    dict : 包含文件详细信息的字典
    """
    data = np.load(filename, allow_pickle=True)
    
    if multiplier_type.upper() == 'R':
        info = {
            'model_name': data['model_name'].item() if hasattr(data['model_name'], 'item') else data['model_name'],
            'N_lens': data['N_lens'],
            'R': data['R'],
            'lgT_obs': data['lgT_obs'],
            'available_multipliers': get_available_multipliers(filename, 'R'),
            'multiplier_type': 'R',
            'all_keys': list(data.keys())
        }
    elif multiplier_type.upper() == 'T':
        info = {
            'model_name': data['model_name'].item() if hasattr(data['model_name'], 'item') else data['model_name'],
            'multiplier_R': data['multiplier_R'],
            'multipliers_T': data['multipliers_T'],
            'available_multipliers': get_available_multipliers(filename, 'T'),
            'multiplier_type': 'T',
            'all_keys': list(data.keys())
        }
    else:
        raise ValueError("multiplier_type must be 'R' or 'T'")
    
    return info


# 在您的主程序中使用
if __name__ == "__main__":
    h5_filename = 'data/lensing_analysis_data/lensing_analysis_results.h5'
    
    # 示例1：处理R倍数数据
    print("="*50)
    print("处理R倍数数据示例")
    print("="*50)
    
    # 运行R倍数的采样分析
    combined_file_R = run_sampling_analysis_R(
        h5_filename=h5_filename,
        model_name='model_1',
        multipliers_R=[1, 0.2, 0.1, 0.02]
    )
    
    # 使用统一函数加载R倍数数据
    file_info_R = get_sampling_data_info(combined_file_R, 'R')
    print(f"模型名称: {file_info_R['model_name']}")
    print(f"透镜数量: {file_info_R['N_lens']}")
    print(f"发生率 R: {file_info_R['R']:.4f}")
    
    # 获取可用的R倍数
    available_multipliers_R = get_available_multipliers(combined_file_R, 'R')
    print(f"可用的R倍数: {available_multipliers_R}")
    
    # 加载特定R倍数的数据
    multiplier_R_to_load = 0.2
    loaded_data_R = load_sampling_data(combined_file_R, multiplier_R_to_load, 'R')
    
    print(f"\n加载R倍数 {multiplier_R_to_load} 的数据:")
    print(f"样本数量: {loaded_data_R['n_samples']}")
    print(f"发生率: {loaded_data_R['R_samples']}")
    print(f"KS统计量: {loaded_data_R['ks_statistic']:.6f}")
    print(f"样本范围 (log10): {loaded_data_R['samples_lg_dt'].min():.2f} 到 {loaded_data_R['samples_lg_dt'].max():.2f}")
    
    # 示例2：处理T倍数数据
    print("\n" + "="*50)
    print("处理T倍数数据示例")
    print("="*50)
    
    # 运行T倍数的采样分析
    combined_file_T = run_sampling_analysis_T(
        h5_filename=h5_filename,
        model_name='model_1',
        multipliers_T=[1, 0.5, 0.1]
    )
    
    # 使用统一函数加载T倍数数据
    file_info_T = get_sampling_data_info(combined_file_T, 'T')
    print(f"模型名称: {file_info_T['model_name']}")
    print(f"固定R倍数: {file_info_T['multiplier_R']}")
    
    # 获取可用的T倍数
    available_multipliers_T = get_available_multipliers(combined_file_T, 'T')
    print(f"可用的T倍数: {available_multipliers_T}")
    
    # 加载特定T倍数的数据
    multiplier_T_to_load = 1
    loaded_data_T = load_sampling_data(combined_file_T, multiplier_T_to_load, 'T')
    
    print(f"\n加载T倍数 {multiplier_T_to_load} 的数据:")
    print(f"样本数量: {loaded_data_T['n_samples']}")
    print(f"KS统计量: {loaded_data_T['ks_statistic']:.6f}")
    print(f"观测时间: {10**loaded_data_T['lgT_obs']/365/24:.1f} 年")
    print(f"模型名称: {loaded_data_T['model_name']}")
    print(f"透镜数量: {loaded_data_T['N_lens']}")