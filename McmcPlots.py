#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 14:54:30 2025

@author: ubuntu
"""

import numpy as np
from modules import compute_interpolated_data
import emcee
import matplotlib.pyplot as plt
import corner
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.special import gammaln
from FiducialPlots import load_sampling_data



# 防止一直调用浪费时间，后续可多文件处理此信息
h5_filename='data/lensing_analysis_data/lensing_analysis_results.h5'
interpolated_data = compute_interpolated_data(h5_filename)


def get_model_interpolation_data(interpolated_data, model_name, lg_fpbh, dt_new):
    """
    从插值数据中获取指定模型和参数下的 p_lg_dt 和 N_lens
    
    参数:
    ----------
    interpolated_data : dict
        包含所有模型插值数据的字典
    model_name : str
        模型名称，如 'model_1', 'model_2' 等
    lg_fpbh : float or array-like
        f_pbh 的对数值
    dt_new : array-like
        时间延迟数据点
        
    返回:
    -------
    p_lg_dt_interp : array-like
        插值后的概率密度函数值
    N_lens_interp : float
        插值后的透镜事件数
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
            np.full_like(dt_new, lg_fpbh),
            np.log10(dt_new)
        ))
        p_lg_dt_interp = interp_p_lg_dt(points)
        N_lens_interp = float(interp_N_lens(lg_fpbh))
    else:
        # 多个 f_pbh 值
        p_lg_dt_interp = []
        N_lens_interp = []
        for f_val in lg_fpbh:
            points = np.column_stack((
                np.full_like(dt_new, f_val),
                np.log10(dt_new)
            ))
            p_lg_dt_interp.append(interp_p_lg_dt(points))
            N_lens_interp.append(float(interp_N_lens(f_val)))
        
        p_lg_dt_interp = np.array(p_lg_dt_interp)
        N_lens_interp = np.array(N_lens_interp)
    
    return p_lg_dt_interp, N_lens_interp


# 给出不同模型在模拟数据之下的后验分布用于后续的MCM计算参数估计
def posterior_distribution(lg_fpbh, model_name='model_2'):
    """
    用其他模型去拟合模拟的数据
    """
    # 使用默认的密集网格得到归一化的截断插值数据
    
    
    # 使用新的插值函数获取 p_lg_dt 和 N_lens
    p_lg_dt_interp, N_lens_interp = get_model_interpolation_data(
        interpolated_data, model_name, lg_fpbh, samples_dt
    )
    
    # 检查插值结果的有效性
    if np.any(p_lg_dt_interp <= 0) or N_lens_interp <= 0:
        return -np.inf
    
    if -6 <= lg_fpbh <= -2:
        n_lens = len(samples_lg_dt)
        
        # 使用对数形式计算，避免数值问题
        # 修正：使用gammaln而不是斯特林近似，更精确
        log_poisson_term = n_lens * np.log(N_lens_interp) - N_lens_interp
        log_factorial_term = gammaln(n_lens + 1)  # ln(n_lens!)
        log_likelihood_term = np.sum(np.log(p_lg_dt_interp))
        
        log_posterior_dt = log_likelihood_term
        log_posterior_n = log_poisson_term - log_factorial_term
        
        return log_posterior_dt+log_posterior_n
    else:
        return -np.inf
    
#print(posterior_distribution(-3))


def run_emcee_sampler(n_walkers=50, n_steps=5000, n_burnin=1000):
    """
    Run EMCEE sampler to estimate posterior distribution of lg_fpbh
    """
    # Define log probability function (for emcee)
    def log_probability(theta):
        lg_fpbh = theta[0]
        return posterior_distribution(lg_fpbh)
    
    # Initialize sampler
    ndim = 1  # parameter dimension
    initial_guess = -4.0  # initial guess for lg_fpbh
    
    # Randomly initialize walker positions around initial guess
    starting_guesses = initial_guess + 1e-2 * np.random.randn(n_walkers, ndim)
    
    # Create sampler without pool
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability)
    
    # Run MCMC sampling with progress bar
    print("Starting MCMC sampling...")
    sampler.run_mcmc(starting_guesses, n_steps, progress=True)
    print("Sampling completed!")
    
    # Get sampling chain
    samples = sampler.get_chain()
    
    return sampler, samples

def analyze_results(sampler, n_burnin=1000):
    """
    Analyze MCMC results and plot posterior distribution
    """
    # Get samples after removing burn-in period
    flat_samples = sampler.get_chain(discard=n_burnin, thin=15, flat=True)
    
    # Calculate statistics
    lg_fpbh_mean = np.mean(flat_samples[:, 0])
    lg_fpbh_std = np.std(flat_samples[:, 0])
    lg_fpbh_median = np.median(flat_samples[:, 0])
    
    # Calculate confidence intervals
    lg_fpbh_lower = np.percentile(flat_samples[:, 0], 16)
    lg_fpbh_upper = np.percentile(flat_samples[:, 0], 84)
    
    print(f"Posterior distribution statistics:")
    print(f"  Mean: lg_fpbh = {lg_fpbh_mean:.3f} ± {lg_fpbh_std:.3f}")
    print(f"  Median: lg_fpbh = {lg_fpbh_median:.3f}")
    print(f"  68% Confidence Interval: [{lg_fpbh_lower:.3f}, {lg_fpbh_upper:.3f}]")
    
    # 修复绘图部分 - 使用正确的子图创建方式
    fig = plt.figure(figsize=(10, 8))
    
    # 1. Sampling chain trace plot
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(sampler.get_chain()[:, :, 0], alpha=0.7)
    ax1.set_ylabel('lg_fpbh')
    ax1.set_xlabel('Step')
    ax1.set_title('MCMC Sampling Chain')
    ax1.axvline(n_burnin, color='red', linestyle='--', alpha=0.7, label='Burn-in end')
    ax1.legend()
    
    # 2. Posterior distribution histogram
    ax2 = plt.subplot(2, 1, 2)
    ax2.hist(flat_samples[:, 0], bins=50, density=True, alpha=0.7, color='blue')
    ax2.axvline(lg_fpbh_median, color='red', linestyle='-', label=f'Median: {lg_fpbh_median:.3f}')
    ax2.axvline(lg_fpbh_lower, color='red', linestyle='--', alpha=0.7, label='68% CI')
    ax2.axvline(lg_fpbh_upper, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('lg_fpbh')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('lg_fpbh Posterior Distribution')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('Plots/lg_fpbh_posterior.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot corner plot
    fig = corner.corner(flat_samples, bins=20, show_titles=True, \
                    title_fmt=".2f", title_kwargs={"fontsize":18}, plot_datapoints= False,\
                        smooth=2.0,smooth1d=2.0,plot_density=True, color='green',levels=(0.6826, 0.9544),\
                            labels=[r'$\log_{10} f_{\mathrm{pbh}}$'], label_kwargs = {"fontsize":28})

    plt.savefig('Plots/lg_fpbh_corner.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return flat_samples

# Main program
if __name__ == "__main__":
    # Ensure sample data is loaded
    try:
        # Load sample data (adjust based on your actual data file)
        data = np.load('data/simulation_data/time_delay_samples_CDM.npz')
        samples_lg_dt = data['samples_lg_dt']
        samples_dt = data['samples_dt']
        print(f"Loaded {len(samples_lg_dt)} samples")
    except FileNotFoundError:
        print("Error: Sample data file not found")
        exit(1)
    
    # Run MCMC sampling with multiprocessing
    sampler, samples = run_emcee_sampler(
        n_walkers=10, 
        n_steps=10000, 
        n_burnin=500
    )
    
    # Analyze results
    flat_samples = analyze_results(sampler, n_burnin=500)
    
    # Save results
    np.savez('data/mcmc_data/mcmc_results.npz', 
             flat_samples=flat_samples,
             acceptance_rate=np.mean(sampler.acceptance_fraction))
    
    print(f"\nMean acceptance rate: {np.mean(sampler.acceptance_fraction):.3f}")

