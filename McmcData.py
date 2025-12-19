#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 14:54:30 2025

@author: ubuntu
"""

import numpy as np
import emcee
import os
from scipy.special import gammaln
from scipy.interpolate import RegularGridInterpolator, interp1d
from FiducialPlots import get_available_multipliers, load_sampling_data
from modules import compute_interpolated_data
import h5py

def get_model_interpolation_data(interpolated_data, model_name, lg_fpbh, samples_dt):
    """
    从插值数据中获取指定模型和参数下的 p_lg_dt 和 N_lens
    此时都是在R=5e5条件下的计算结果
    """
    # 访问特定模型的数据
    model_data = interpolated_data['models'][model_name]
    fpbh_list = interpolated_data['new_f_pbh_values']
    dt_list = interpolated_data['new_a_values']
    
    # 原始数据网格
    dt_expanded = np.tile(dt_list, (len(fpbh_list), 1))
    p_lg_dt = model_data['integrand_dense'] * dt_expanded * np.log(10)
    
    N_lens = model_data['N_lens_multiplied']
    
    
    # p_lg_dt 是二维插值 (f_pbh, dt)
    interp_p_lg_dt = RegularGridInterpolator(
        (np.log10(fpbh_list), np.log10(dt_list)), 
        p_lg_dt,
        bounds_error=False, 
        fill_value=0.0
    )
    
    # N_lens 是一维插值 (f_pbh)
    interp_N_lens = interp1d(
        np.log10(fpbh_list), 
        N_lens, 
        bounds_error=False, 
        fill_value=0.0
    )
    
    # 对输入参数进行插值
    if np.isscalar(lg_fpbh):
        # 单个 f_pbh 值
        points = np.column_stack((
            np.full_like(samples_dt, lg_fpbh),
            np.log10(samples_dt)
        ))
        p_lg_dt_interp = interp_p_lg_dt(points)
        N_lens_interp = float(interp_N_lens(lg_fpbh))
    else:
        # 多个 f_pbh 值
        p_lg_dt_interp = []
        N_lens_interp = []
        for f_val in lg_fpbh:
            points = np.column_stack((
                np.full_like(samples_dt, f_val),
                np.log10(samples_dt)
            ))
            p_lg_dt_interp.append(interp_p_lg_dt(points))
            N_lens_interp.append(float(interp_N_lens(f_val)))
        
        p_lg_dt_interp = np.array(p_lg_dt_interp)
        N_lens_interp = np.array(N_lens_interp)
    
    return p_lg_dt_interp, N_lens_interp


def posterior_distribution(lg_fpbh, samples_dt, interpolated_data, model_name, m_R):
    """
    用其他模型去拟合模拟的数据
    """
    # 使用新的插值函数获取 p_lg_dt 和 N_lens
    p_lg_dt_interp, N_lens_interp = get_model_interpolation_data(
        interpolated_data, model_name, lg_fpbh, samples_dt
    )
    N_lens_interp = N_lens_interp*m_R
    
    # 检查插值结果的有效性
    if np.any(p_lg_dt_interp <= 0) or N_lens_interp <= 0:
        return -np.inf
    
    if -5 <= lg_fpbh <= -2:
        n_lens = len(samples_dt)
        
        # 使用对数形式计算，避免数值问题
        log_poisson_term = n_lens * np.log(N_lens_interp) - N_lens_interp
        log_factorial_term = gammaln(n_lens + 1)  # ln(n_lens!)
        
        # 计算对数似然项
        log_posterior_dt = np.sum(np.log(p_lg_dt_interp))
        
        log_posterior_n = log_poisson_term - log_factorial_term
        
        return log_posterior_dt + log_posterior_n
    else:
        return -np.inf



def run_emcee_sampler(samples_dt, interpolated_data, model_name, m_R, 
                     n_walkers=50, n_steps=10000, n_burnin=None, thin=None,
                     upper_limit_only=True):
    """
    Run EMCEE sampler to estimate posterior distribution of lg_fpbh
    
    Parameters:
    - upper_limit_only: if True, handles cases where posterior doesn't converge
                       at low f_pbh (for upper limit estimation)
    """
    
    # Input validation
    if len(samples_dt) == 0:
        raise ValueError("samples_dt cannot be empty")
    
    if model_name not in interpolated_data['models']:
        raise ValueError(f"Model {model_name} not found in interpolated_data")
    
    # Define log probability function
    def log_probability(theta):
        lg_fpbh = theta[0]
        return posterior_distribution(lg_fpbh, samples_dt, interpolated_data, model_name, m_R)
    
    # Initialize sampler
    ndim = 1  # parameter dimension
    
    # For upper limit estimation, initialize walkers across the plausible range
    # Focus more on the upper end where we expect the limit to be
    prior_range = [-5, -2]  # Wider range for upper limit estimation
    starting_guesses = np.random.uniform(
        prior_range[0], prior_range[1], (n_walkers, ndim)
    )
    
    # Create sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, 
        ndim, 
        log_probability,
        moves=emcee.moves.StretchMove(a=2.0)
    )
    
    # Run MCMC sampling with progress bar
    print("Starting MCMC sampling...")
    state = sampler.run_mcmc(starting_guesses, n_steps, progress=True)
    print("Sampling completed!")
    
    # Get sampling chain
    samples = sampler.get_chain()
    
    # For upper limit estimation, handle autocorrelation time calculation carefully
    tau_avg = None
    try:
        # Try to calculate autocorrelation time with conservative settings
        tau = sampler.get_autocorr_time(discard=n_steps//2, quiet=True, c=10)
        tau_avg = np.mean(tau)
        
        # Check if we got reasonable values
        if np.any(np.isnan(tau)) or np.any(tau > n_steps/2):
            print("Autocorrelation time estimation failed (NaN or too large)")
            tau_avg = None
        else:
            print(f"Average autocorrelation time: {tau_avg:.2f}")
            
    except (emcee.autocorr.AutocorrError, ValueError, RuntimeWarning) as e:
        print(f"Could not estimate autocorrelation time: {e}")
        tau_avg = None
    
    # Set thinning parameter
    if thin is None:
        if tau_avg is not None and tau_avg > 0:
            thin = max(1, int(0.5 * tau_avg))
            print(f"Auto-set thinning to: {thin}")
        else:
            thin = 1
            print("Using thin=1 (no thinning) due to autocorrelation time estimation issues")
    
    # Set burn-in period
    if n_burnin is None:
        if tau_avg is not None and tau_avg > 0:
            n_burnin = min(n_steps - 100, int(5 * tau_avg))  # More conservative for upper limits
        else:
            # For upper limit estimation without good autocorrelation time,
            # use a fixed fraction but check convergence manually
            n_burnin = n_steps // 2
        print(f"Auto-set burn-in to: {n_burnin}")
    
    # Apply thinning and get flat samples
    flat_samples = sampler.get_chain(discard=n_burnin, thin=thin, flat=True)
    
    return sampler, samples, flat_samples

def run_mcmc_for_all_multipliers(h5_filename, model_name,
                                multipliers_R=None, multipliers_T=None,
                                n_walkers=50, n_steps=5000, n_burnin=1000):
    """
    为所有倍数运行MCMC采样并保存结果
    """
    
    # 确保输出目录存在
    os.makedirs('data/mcmc_data', exist_ok=True)
    
    # 分别存储R和T类型的结果
    results_summary_R = {}
    results_summary_T = {}
    
    # 处理R倍数
    if multipliers_R is not None:
        print("="*60)
        print("Processing R multipliers")
        print("="*60)
        
        # 加载R倍数数据文件
        R_filename = 'data/simulation_data/time_delay_samples_CDM_all_multipliers_R.npz'
        if not os.path.exists(R_filename):
            print(f"Warning: R multiplier file {R_filename} not found. Skipping R multipliers.")
        else:
            # 获取可用的R倍数
            available_R_multipliers = get_available_multipliers(R_filename, 'R')
            print(f"Available R multipliers: {available_R_multipliers}")
            
            # 只处理指定的倍数
            multipliers_to_process = [m for m in multipliers_R if m in available_R_multipliers]
            print(f"Processing R multipliers: {multipliers_to_process}")
            
            # 创建HDF5文件用于保存所有R倍数的结果 - 包含模型名称
            h5_R_filename = f'data/mcmc_data/mcmc_results_R_{model_name}.h5'
            
            for multiplier in multipliers_to_process:
                print(f"\n--- Processing R multiplier {multiplier} ---")
                
                # 加载样本数据
                try:
                    loaded_data = load_sampling_data(R_filename, multiplier, 'R')
                    samples_dt = loaded_data['samples_dt']
                    
                    # 获取插值数据 - 使用默认观测时间 (multiplier_T=1)
                    interpolated_data = compute_interpolated_data(h5_filename, multipliers_T=1)
                    
                    # 运行MCMC
                    sampler, samples, flat_samples = run_emcee_sampler(
                        samples_dt=samples_dt,
                        interpolated_data=interpolated_data,
                        model_name=model_name,
                        m_R = multiplier,
                        n_walkers=n_walkers,
                        n_steps=n_steps,
                        n_burnin=n_burnin
                    )
                    
                    # 保存结果到HDF5文件
                    multiplier_str = str(multiplier).replace('.', '_')
                    
                    with h5py.File(h5_R_filename, 'a') as f:
                        group_name = f'multiplier_{multiplier_str}'
                        if group_name in f:
                            del f[group_name]  # 删除已存在的组
                        
                        group = f.create_group(group_name)
                        
                        # 保存MCMC链数据
                        group.create_dataset('flat_samples', data=flat_samples)
                        group.create_dataset('samples', data=samples)
                        group.create_dataset('sampler_chain', data=sampler.chain)
                        group.create_dataset('acceptance_fraction', data=sampler.acceptance_fraction)
                        
                        # 保存元数据作为属性
                        group.attrs['acceptance_rate'] = np.mean(sampler.acceptance_fraction)
                        group.attrs['multiplier'] = multiplier
                        group.attrs['multiplier_type'] = 'R'
                        group.attrs['model_name'] = model_name
                        group.attrs['n_samples'] = len(samples_dt)
                        group.attrs['n_walkers'] = n_walkers
                        group.attrs['n_steps'] = n_steps
                        group.attrs['n_burnin'] = n_burnin
                        
                        # 保存统计摘要
                        group.attrs['mean_lg_fpbh'] = np.mean(flat_samples)
                        group.attrs['std_lg_fpbh'] = np.std(flat_samples)
                        group.attrs['median_lg_fpbh'] = np.median(flat_samples)
                        group.attrs['q5_lg_fpbh'] = np.percentile(flat_samples, 5)
                        group.attrs['q95_lg_fpbh'] = np.percentile(flat_samples, 95)
                        group.attrs['n_effective_samples'] = len(flat_samples)
                    
                    # 记录结果摘要
                    results_summary_R[f'multiplier_{multiplier_str}'] = {
                        'acceptance_rate': np.mean(sampler.acceptance_fraction),
                        'mean_lg_fpbh': np.mean(flat_samples),
                        'std_lg_fpbh': np.std(flat_samples),
                        'median_lg_fpbh': np.median(flat_samples),
                        'q5_lg_fpbh': np.percentile(flat_samples, 5),
                        'q95_lg_fpbh': np.percentile(flat_samples, 95),
                        'n_effective_samples': len(flat_samples),
                        'multiplier': multiplier,
                        'multiplier_type': 'R',
                        'n_walkers': n_walkers,
                        'n_steps': n_steps,
                        'n_burnin': n_burnin
                    }
                    
                    print(f"Results saved to HDF5 group: {group_name}")
                    print(f"Acceptance rate: {np.mean(sampler.acceptance_fraction):.3f}")
                    print(f"Mean lg_fpbh: {np.mean(flat_samples):.4f} ± {np.std(flat_samples):.4f}")
                    
                except Exception as e:
                    print(f"Error processing R multiplier {multiplier}: {e}")
    
    # 处理T倍数
    if multipliers_T is not None:
        print("\n" + "="*60)
        print("Processing T multipliers")
        print("="*60)
        
        # 加载T倍数数据文件
        T_filename = 'data/simulation_data/time_delay_samples_CDM_all_multipliers_T.npz'
        if not os.path.exists(T_filename):
            print(f"Warning: T multiplier file {T_filename} not found. Skipping T multipliers.")
        else:
            # 获取可用的T倍数
            available_T_multipliers = get_available_multipliers(T_filename, 'T')
            print(f"Available T multipliers: {available_T_multipliers}")
            
            # 只处理指定的倍数
            multipliers_to_process = [m for m in multipliers_T if m in available_T_multipliers]
            print(f"Processing T multipliers: {multipliers_to_process}")
            
            # 创建HDF5文件用于保存所有T倍数的结果 - 包含模型名称
            h5_T_filename = f'data/mcmc_data/mcmc_results_T_{model_name}.h5'
            
            for multiplier in multipliers_to_process:
                print(f"\n--- Processing T multiplier {multiplier} ---")
                
                # 加载样本数据
                try:
                    loaded_data = load_sampling_data(T_filename, multiplier, 'T')
                    samples_dt = loaded_data['samples_dt']
                    m_R_T = loaded_data['multiplier_R']
                    #print(m_R_T)
                    
                    # 获取插值数据 - 使用指定的观测时间倍数
                    interpolated_data = compute_interpolated_data(h5_filename, multipliers_T=multiplier)
                    
                    # 运行MCMC
                    sampler, samples, flat_samples = run_emcee_sampler(
                        samples_dt=samples_dt,
                        interpolated_data=interpolated_data,
                        model_name=model_name,
                        m_R = m_R_T,
                        n_walkers=n_walkers,
                        n_steps=n_steps,
                        n_burnin=n_burnin
                    )
                    
                    # 保存结果到HDF5文件
                    multiplier_str = str(multiplier).replace('.', '_')
                    
                    with h5py.File(h5_T_filename, 'a') as f:
                        group_name = f'multiplier_{multiplier_str}'
                        if group_name in f:
                            del f[group_name]  # 删除已存在的组
                        
                        group = f.create_group(group_name)
                        
                        # 保存MCMC链数据
                        group.create_dataset('flat_samples', data=flat_samples)
                        group.create_dataset('samples', data=samples)
                        group.create_dataset('sampler_chain', data=sampler.chain)
                        group.create_dataset('acceptance_fraction', data=sampler.acceptance_fraction)
                        
                        # 保存元数据作为属性
                        group.attrs['acceptance_rate'] = np.mean(sampler.acceptance_fraction)
                        group.attrs['multiplier'] = multiplier
                        group.attrs['multiplier_type'] = 'T'
                        group.attrs['model_name'] = model_name
                        group.attrs['n_samples'] = len(samples_dt)
                        group.attrs['n_walkers'] = n_walkers
                        group.attrs['n_steps'] = n_steps
                        group.attrs['n_burnin'] = n_burnin
                        
                        # 保存统计摘要
                        group.attrs['mean_lg_fpbh'] = np.mean(flat_samples)
                        group.attrs['std_lg_fpbh'] = np.std(flat_samples)
                        group.attrs['median_lg_fpbh'] = np.median(flat_samples)
                        group.attrs['q5_lg_fpbh'] = np.percentile(flat_samples, 5)
                        group.attrs['q95_lg_fpbh'] = np.percentile(flat_samples, 95)
                        group.attrs['n_effective_samples'] = len(flat_samples)
                    
                    # 记录结果摘要
                    results_summary_T[f'multiplier_{multiplier_str}'] = {
                        'acceptance_rate': np.mean(sampler.acceptance_fraction),
                        'mean_lg_fpbh': np.mean(flat_samples),
                        'std_lg_fpbh': np.std(flat_samples),
                        'median_lg_fpbh': np.median(flat_samples),
                        'q5_lg_fpbh': np.percentile(flat_samples, 5),
                        'q95_lg_fpbh': np.percentile(flat_samples, 95),
                        'n_effective_samples': len(flat_samples),
                        'multiplier': multiplier,
                        'multiplier_type': 'T',
                        'n_walkers': n_walkers,
                        'n_steps': n_steps,
                        'n_burnin': n_burnin
                    }
                    
                    print(f"Results saved to HDF5 group: {group_name}")
                    print(f"Acceptance rate: {np.mean(sampler.acceptance_fraction):.3f}")
                    print(f"Mean lg_fpbh: {np.mean(flat_samples):.4f} ± {np.std(flat_samples):.4f}")
                    
                except Exception as e:
                    print(f"Error processing T multiplier {multiplier}: {e}")
    
    # 保存汇总摘要到HDF5文件 - 包含模型名称
    summary_filename = f'data/mcmc_data/mcmc_results_summary_{model_name}.h5'
    with h5py.File(summary_filename, 'w') as f:
        # 保存R倍数摘要
        if results_summary_R:
            r_group = f.create_group('R_multipliers')
            for key, summary in results_summary_R.items():
                subgroup = r_group.create_group(key)
                for attr_name, attr_value in summary.items():
                    if isinstance(attr_value, (int, float, str)):
                        subgroup.attrs[attr_name] = attr_value
        
        # 保存T倍数摘要
        if results_summary_T:
            t_group = f.create_group('T_multipliers')
            for key, summary in results_summary_T.items():
                subgroup = t_group.create_group(key)
                for attr_name, attr_value in summary.items():
                    if isinstance(attr_value, (int, float, str)):
                        subgroup.attrs[attr_name] = attr_value
        
        # 保存全局信息
        f.attrs['model_name'] = model_name
        f.attrs['n_walkers'] = n_walkers
        f.attrs['n_steps'] = n_steps
        f.attrs['n_burnin'] = n_burnin
        f.attrs['creation_date'] = np.bytes_(str(np.datetime64('now')))
    
    print(f"\nSummary saved to: {summary_filename}")
    
    # 返回合并的摘要
    combined_summary = {
        'R_multipliers': results_summary_R,
        'T_multipliers': results_summary_T,
        'model_name': model_name,
        'n_walkers': n_walkers,
        'n_steps': n_steps,
        'n_burnin': n_burnin
    }
    
    return combined_summary


# 修改后的加载函数：从HDF5文件加载MCMC结果
def load_mcmc_results(multiplier_type, multiplier, model_name='model_2', results_dir='data/mcmc_data'):
    """
    从HDF5文件加载特定类型和倍数的MCMC结果
    
    Parameters:
    -----------
    multiplier_type : str
        倍数类型，'R' 或 'T'
    multiplier : float
        倍数
    model_name : str
        模型名称
    results_dir : str
        结果目录路径
    
    Returns:
    --------
    dict : 包含MCMC结果的字典
    """
    multiplier_str = str(multiplier).replace('.', '_')
    filename = f'{results_dir}/mcmc_results_{multiplier_type}_{model_name}.h5'
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"MCMC results file not found: {filename}")
    
    result = {}
    
    with h5py.File(filename, 'r') as f:
        group_name = f'multiplier_{multiplier_str}'
        if group_name not in f:
            available_groups = list(f.keys())
            raise KeyError(f"Group '{group_name}' not found in file. Available groups: {available_groups}")
        
        group = f[group_name]
        
        # 加载数据集
        for key in group.keys():
            result[key] = group[key][:]
        
        # 加载属性
        for attr_name in group.attrs.keys():
            result[attr_name] = group.attrs[attr_name]
    
    return result


# 修改后的加载函数：从HDF5文件加载摘要信息
def load_mcmc_summary(model_name='model_2', results_dir='data/mcmc_data'):
    """
    从HDF5文件加载MCMC摘要信息
    
    Parameters:
    -----------
    model_name : str
        模型名称
    results_dir : str
        结果目录路径
    
    Returns:
    --------
    dict : 包含摘要信息的字典
    """
    filename = f'{results_dir}/mcmc_results_summary_{model_name}.h5'
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Summary file not found: {filename}")
    
    summary = {}
    
    with h5py.File(filename, 'r') as f:
        # 加载全局属性
        for attr_name in f.attrs.keys():
            summary[attr_name] = f.attrs[attr_name]
        
        # 加载R倍数摘要
        if 'R_multipliers' in f:
            summary['R_multipliers'] = {}
            r_group = f['R_multipliers']
            for subgroup_name in r_group.keys():
                subgroup = r_group[subgroup_name]
                summary['R_multipliers'][subgroup_name] = {}
                for attr_name in subgroup.attrs.keys():
                    summary['R_multipliers'][subgroup_name][attr_name] = subgroup.attrs[attr_name]
        
        # 加载T倍数摘要
        if 'T_multipliers' in f:
            summary['T_multipliers'] = {}
            t_group = f['T_multipliers']
            for subgroup_name in t_group.keys():
                subgroup = t_group[subgroup_name]
                summary['T_multipliers'][subgroup_name] = {}
                for attr_name in subgroup.attrs.keys():
                    summary['T_multipliers'][subgroup_name][attr_name] = subgroup.attrs[attr_name]
    
    return summary


# 修改后的函数：获取HDF5文件中可用的倍数
def get_available_multipliers_from_h5(multiplier_type, model_name='model_2', results_dir='data/mcmc_data'):
    """
    从HDF5文件获取可用的倍数
    
    Parameters:
    -----------
    multiplier_type : str
        倍数类型，'R' 或 'T'
    model_name : str
        模型名称
    results_dir : str
        结果目录路径
    
    Returns:
    --------
    list : 可用的倍数列表
    """
    filename = f'{results_dir}/mcmc_results_{multiplier_type}_{model_name}.h5'
    
    if not os.path.exists(filename):
        return []
    
    multipliers = []
    
    with h5py.File(filename, 'r') as f:
        for group_name in f.keys():
            if group_name.startswith('multiplier_'):
                multiplier_str = group_name.replace('multiplier_', '')
                if '_' in multiplier_str:
                    multiplier = float(multiplier_str.replace('_', '.'))
                else:
                    multiplier = int(multiplier_str)
                multipliers.append(multiplier)
    
    return sorted(multipliers)



# 主程序
if __name__ == "__main__":
    h5_filename = 'data/lensing_analysis_data/dt/lensing_analysis_results_1e+10_10_dt.h5'
    
    # 运行MCMC分析，处理R和T倍数
    combined_summary = run_mcmc_for_all_multipliers(
        h5_filename=h5_filename,
        model_name='model_3',
        #multipliers_R=[1, 0.2, 0.1, 0.02],  # R倍数
        #multipliers_T=[1, 0.5, 0.2, 0.1],   # T倍数
        multipliers_R=[1],
        multipliers_T=[1],
        n_walkers=20,
        n_steps=10000,
        n_burnin=500
    )
    
    
    # 打印最终摘要
    print("\n" + "="*60)
    print("MCMC Analysis Summary")
    print("="*60)
    
    # 打印R倍数结果
    if 'R_multipliers' in combined_summary and combined_summary['R_multipliers']:
        print("\nR Multipliers:")
        print("-" * 40)
        for key, result in combined_summary['R_multipliers'].items():
            print(f"{key}:")
            print(f"  Acceptance rate: {result['acceptance_rate']:.3f}")
            print(f"  Mean lg_fpbh: {result['mean_lg_fpbh']:.4f} ± {result['std_lg_fpbh']:.4f}")
            print(f"  Median lg_fpbh: {result['median_lg_fpbh']:.4f}")
            print(f"  5-95% quantiles: [{result['q5_lg_fpbh']:.4f}, {result['q95_lg_fpbh']:.4f}]")
            print(f"  Effective samples: {result['n_effective_samples']}")
            print()
    
    # 打印T倍数结果
    if 'T_multipliers' in combined_summary and combined_summary['T_multipliers']:
        print("\nT Multipliers:")
        print("-" * 40)
        for key, result in combined_summary['T_multipliers'].items():
            print(f"{key}:")
            print(f"  Acceptance rate: {result['acceptance_rate']:.3f}")
            print(f"  Mean lg_fpbh: {result['mean_lg_fpbh']:.4f} ± {result['std_lg_fpbh']:.4f}")
            print(f"  Median lg_fpbh: {result['median_lg_fpbh']:.4f}")
            print(f"  5-95% quantiles: [{result['q5_lg_fpbh']:.4f}, {result['q95_lg_fpbh']:.4f}]")
            print(f"  Effective samples: {result['n_effective_samples']}")
            print()
