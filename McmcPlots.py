#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 09:55:25 2025

@author: ubuntu
"""
from McmcData import get_available_multipliers_from_h5, load_mcmc_results,load_mcmc_summary
import matplotlib.pyplot as plt
import numpy as np
import os

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


def plot_fpbh_histograms(model_name, results_dir='data/mcmc_data', 
                        save_plot=True, plot_format='pdf'):
    """
    使用直方图绘制f_pbh后验分布，更好地显示分布的复杂形状
    """
    # 检查文件是否存在模拟的数据与mcmc的数据
    R_filename_s = 'data/simulation_data/time_delay_samples_CDM_all_multipliers_R.npz'
    T_filename_s = 'data/simulation_data/time_delay_samples_CDM_all_multipliers_T.npz'
    R_tot = 5e5
    T_tot = 10.0
    
    R_filename_m = f'{results_dir}/mcmc_results_R_{model_name}.h5'
    T_filename_m = f'{results_dir}/mcmc_results_T_{model_name}.h5'
    
    if not os.path.exists(R_filename_m) and not os.path.exists(T_filename_m):
        print(f"Error: No MCMC results found for model {model_name}")
        return
    
    # 创建图形和子图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 处理R倍数
    if os.path.exists(R_filename_m):
        print("Processing R multipliers for histogram plotting...")
        
        # 获取可用的R倍数
        available_R_multipliers = get_available_multipliers_from_h5('R', model_name, results_dir)
        print(f"Available R multipliers: {available_R_multipliers}")
        
        # 颜色映射
        #colors = plt.cm.viridis(np.linspace(0, 1, len(available_R_multipliers)))
        colors = ['orange', 'green', 'blue', 'red']
        
        for i, multiplier in enumerate(available_R_multipliers):
            try:
                # 加载MCMC结果
                mcmc_data = load_mcmc_results('R', multiplier, model_name, results_dir)
                flat_samples = mcmc_data['flat_samples'].flatten()
                
                # 使用直方图
                color = colors[i]
                R_formatted = format_R_value(R_tot*multiplier)
                label = rf'R = {R_formatted} $\mathrm{{yr}}^{{-1}}$'
                
                # 计算合适的bin数量
                n_bins = min(50, len(flat_samples) // 10)
                n_bins = min(n_bins, 30)  # 至多30个bin
                
                axes[0].hist(flat_samples, bins=n_bins, density=True, alpha=0.5, 
                            color=color, label=label, histtype='step', linewidth=2)
                
                # 标记关键统计量
                median = np.median(flat_samples)
                q5 = np.percentile(flat_samples, 5)
                q95 = np.percentile(flat_samples, 95)
                
                #axes[0].axvline(median, color=color, linestyle='-', alpha=0.8, linewidth=1.5)
                #axes[0].axvline(q5, color=color, linestyle='--', alpha=0.6, linewidth=1)
                axes[0].axvline(q95, color=color, linestyle='--', alpha=1, linewidth=2)
                
            except Exception as e:
                print(f"Error processing R multiplier {multiplier} for histogram: {e}")
        
        # 设置R倍数子图属性
        axes[0].set_xlabel(r'$\log(f_{\mathrm{PBH}})$', fontsize=16)
        axes[0].set_ylabel('Probability Density', fontsize=16)
        axes[0].set_xlim(-5, -2.5)
        axes[0].tick_params(axis='both', which='major', labelsize=16)
        axes[0].legend(fontsize=15)
        axes[0].grid(True, alpha=0.3)
        
    # 处理T倍数
    if os.path.exists(T_filename_m):
        print("Processing T multipliers for histogram plotting...")
        
        # 获取可用的T倍数
        available_T_multipliers = get_available_multipliers_from_h5('T', model_name, results_dir)
        print(f"Available T multipliers: {available_T_multipliers}")
        
        # 颜色映射
        #colors = plt.cm.plasma(np.linspace(0, 1, len(available_T_multipliers)))
        colors = ['orange', 'green', 'blue', 'red']
        
        for i, multiplier in enumerate(available_T_multipliers):
            try:
                # 加载MCMC结果
                mcmc_data = load_mcmc_results('T', multiplier, model_name, results_dir)
                flat_samples = mcmc_data['flat_samples'].flatten()
                
                # 使用直方图
                color = colors[i]
                T_formatted = T_tot*multiplier
                label = rf'$T_{{\rm obs}}$ = {T_formatted} $\rm yrs$'
                
                # 计算合适的bin数量
                n_bins = min(50, len(flat_samples) // 10)
                n_bins = min(n_bins, 30)  # 至多30个bin
                
                axes[1].hist(flat_samples, bins=n_bins, density=True, alpha=0.5, 
                            color=color, label=label, histtype='step', linewidth=2)
                
                # 标记关键统计量
                median = np.median(flat_samples)
                q5 = np.percentile(flat_samples, 5)
                q95 = np.percentile(flat_samples, 95)
                
                #axes[1].axvline(median, color=color, linestyle='-', alpha=0.8, linewidth=1.5)
                #axes[1].axvline(q5, color=color, linestyle='--', alpha=0.6, linewidth=1)
                axes[1].axvline(q95, color=color, linestyle='--', alpha=1, linewidth=2)
                
            except Exception as e:
                print(f"Error processing T multiplier {multiplier} for histogram: {e}")
        
        # 设置T倍数子图属性
        axes[1].set_xlabel(r'$\log(f_{\mathrm{PBH}})$', fontsize=16)
        axes[1].set_ylabel('Probability Density', fontsize=16)
        axes[1].set_xlim(-5, -2.5)
        axes[1].tick_params(axis='both', which='major', labelsize=16)
        axes[1].legend(fontsize=15)
        axes[1].grid(True, alpha=0.3)

    
    # 调整布局
    plt.tight_layout()
    
    
    # 保存图形
    if save_plot:
        plot_dir = 'Plots'
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = f'{plot_dir}/fpbh_histograms_{model_name}.{plot_format}'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Histogram plot saved to: {plot_filename}")
    
    plt.show()
    
    return fig, axes


def plot_fpbh_credible_intervals(model_name, results_dir='data/mcmc_data', 
                                save_plot=True, plot_format='pdf'):
    """
    绘制可信区间图，更好地显示参数估计的不确定性
    """
    
    # 检查文件是否存在模拟的数据与mcmc的数据
    R_filename_s = 'data/simulation_data/time_delay_samples_CDM_all_multipliers_R.npz'
    T_filename_s = 'data/simulation_data/time_delay_samples_CDM_all_multipliers_T.npz'
    R_tot = 5e5
    T_tot = 10.0
    
    R_filename = f'{results_dir}/mcmc_results_R_{model_name}.h5'
    T_filename = f'{results_dir}/mcmc_results_T_{model_name}.h5'
    
    if not os.path.exists(R_filename) and not os.path.exists(T_filename):
        print(f"Error: No MCMC results found for model {model_name}")
        return
    
    # 创建图形和子图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    def plot_credible_intervals(data, ax, color, label, position):
        """绘制可信区间"""
        median = np.median(data)
        q16 = np.percentile(data, 16)
        q84 = np.percentile(data, 84)
        q5 = np.percentile(data, 5)
        q95 = np.percentile(data, 95)
        
        # 绘制68%和90%可信区间
        ax.errorbar(position, median, 
                   yerr=[[median - q16], [q84 - median]], 
                   fmt='o', color=color, capsize=5, capthick=2, 
                   markersize=8, label=label)
        
        ax.errorbar(position, median, 
                   yerr=[[median - q5], [q95 - median]], 
                   fmt='none', color=color, capsize=3, capthick=1, 
                   alpha=0.7, elinewidth=1)
    
    # 处理R倍数
    if os.path.exists(R_filename):
        print("Processing R multipliers for credible intervals...")
        
        # 获取可用的R倍数
        available_R_multipliers = get_available_multipliers_from_h5('R', model_name, results_dir)
        print(f"Available R multipliers: {available_R_multipliers}")
        
        # 颜色映射
        #colors = plt.cm.viridis(np.linspace(0, 1, len(available_R_multipliers)))
        colors = ['orange', 'green', 'blue', 'red']
        
        for i, multiplier in enumerate(available_R_multipliers):
            try:
                # 加载MCMC结果
                mcmc_data = load_mcmc_results('R', multiplier, model_name, results_dir)
                flat_samples = mcmc_data['flat_samples'].flatten()
                
                color = colors[i]
                R_formatted = format_R_value(R_tot*multiplier)
                label = rf'R = {R_formatted} $\mathrm{{yr}}^{{-1}}$'
                
                plot_credible_intervals(flat_samples, axes[0], color, label, i)
                
            except Exception as e:
                print(f"Error processing R multiplier {multiplier} for credible intervals: {e}")
        
        # 设置R倍数子图属性
        axes[0].set_xlabel('Index', fontsize=16)
        axes[0].set_ylabel(r'$\log(f_{\mathrm{PBH}})$', fontsize=16)
        #axes[0].set_xticks(range(len(available_R_multipliers)))
        #axes[0].set_xticklabels([f'R={m}' for m in available_R_multipliers], rotation=45)
        axes[0].legend(fontsize=15)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='both', which='major', labelsize=18)
    
    # 处理T倍数
    if os.path.exists(T_filename):
        print("Processing T multipliers for credible intervals...")
        
        # 获取可用的T倍数
        available_T_multipliers = get_available_multipliers_from_h5('T', model_name, results_dir)
        print(f"Available T multipliers: {available_T_multipliers}")
        
        # 颜色映射
        #colors = plt.cm.plasma(np.linspace(0, 1, len(available_T_multipliers)))
        colors = ['orange', 'green', 'blue', 'red']
        
        for i, multiplier in enumerate(available_T_multipliers):
            try:
                # 加载MCMC结果
                mcmc_data = load_mcmc_results('T', multiplier, model_name, results_dir)
                flat_samples = mcmc_data['flat_samples'].flatten()
                
                color = colors[i]
                T_formatted = T_tot*multiplier
                label = rf'$T_{{\rm obs}}$ = {T_formatted} $\rm yrs$'
                
                plot_credible_intervals(flat_samples, axes[1], color, label, i)
                
            except Exception as e:
                print(f"Error processing T multiplier {multiplier} for credible intervals: {e}")
        
        # 设置T倍数子图属性
        axes[1].set_xlabel('Index', fontsize=16)
        axes[1].set_ylabel(r'$\log(f_{\mathrm{PBH}})$', fontsize=16)
        #axes[1].set_xticks(range(len(available_T_multipliers)))
        #axes[1].set_xticklabels([f'T={m}' for m in available_T_multipliers], rotation=45)
        axes[1].legend(fontsize=15)
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='both', which='major', labelsize=18)
    
    # 调整布局
    plt.tight_layout()
    
    
    # 保存图形
    if save_plot:
        plot_dir = 'Plots'
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = f'{plot_dir}/fpbh_credible_intervals_{model_name}.{plot_format}'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Credible intervals plot saved to: {plot_filename}")

    plt.show()
    
    return fig, axes


def create_enhanced_fpbh_analysis(model_name, results_dir='data/mcmc_data'):
    """
    创建增强的f_pbh分析，使用多种可视化方法
    """
    
    print("="*60)
    print(f"Enhanced f_PBH Analysis for {model_name}")
    print("="*60)
    
    # 加载摘要信息
    try:
        summary = load_mcmc_summary(model_name, results_dir)
        print(f"Model: {summary.get('model_name', 'Unknown')}")
        print(f"Number of walkers: {summary.get('n_walkers', 'Unknown')}")
        print(f"Number of steps: {summary.get('n_steps', 'Unknown')}")
        print(f"Burn-in steps: {summary.get('n_burnin', 'Unknown')}")
    except Exception as e:
        print(f"Error loading summary: {e}")
    
    # 绘制多种可视化图形
    print("\n" + "-"*40)
    print("Creating histogram plots...")
    plot_fpbh_histograms(model_name, results_dir)
    
    print("\n" + "-"*40)
    print("Creating credible interval plots...")
    plot_fpbh_credible_intervals(model_name, results_dir)
    
    # 打印详细统计信息
    print("\n" + "-"*40)
    print("Detailed Statistics:")
    print("-"*40)
    
    # R倍数统计
    R_filename = f'{results_dir}/mcmc_results_R_{model_name}.h5'
    if os.path.exists(R_filename):
        print("\nR Multipliers:")
        available_R_multipliers = get_available_multipliers_from_h5('R', model_name, results_dir)
        for multiplier in available_R_multipliers:
            try:
                mcmc_data = load_mcmc_results('R', multiplier, model_name, results_dir)
                flat_samples = mcmc_data['flat_samples'].flatten()
                
                median = np.median(flat_samples)
                q5 = np.percentile(flat_samples, 5)
                q95 = np.percentile(flat_samples, 95)
                mean = np.mean(flat_samples)
                std = np.std(flat_samples)
                
                print(f"  R={multiplier}:")
                print(f"    Mean: {mean:.4f}")
                print(f"    Std: {std:.4f}")
                print(f"    Median: {median:.4f}")
                print(f"    90% CI: [{q5:.4f}, {q95:.4f}]")
                print(f"    Acceptance rate: {mcmc_data['acceptance_rate']:.3f}")
                print(f"    Effective samples: {len(flat_samples)}")
                
            except Exception as e:
                print(f"  Error processing R multiplier {multiplier}: {e}")
    
    # T倍数统计
    T_filename = f'{results_dir}/mcmc_results_T_{model_name}.h5'
    if os.path.exists(T_filename):
        print("\nT Multipliers:")
        available_T_multipliers = get_available_multipliers_from_h5('T', model_name, results_dir)
        for multiplier in available_T_multipliers:
            try:
                mcmc_data = load_mcmc_results('T', multiplier, model_name, results_dir)
                flat_samples = mcmc_data['flat_samples'].flatten()
                
                median = np.median(flat_samples)
                q5 = np.percentile(flat_samples, 5)
                q95 = np.percentile(flat_samples, 95)
                mean = np.mean(flat_samples)
                std = np.std(flat_samples)
                
                print(f"  T={multiplier}:")
                print(f"    Mean: {mean:.4f}")
                print(f"    Std: {std:.4f}")
                print(f"    Median: {median:.4f}")
                print(f"    90% CI: [{q5:.4f}, {q95:.4f}]")
                print(f"    Acceptance rate: {mcmc_data['acceptance_rate']:.3f}")
                print(f"    Effective samples: {len(flat_samples)}")
                
            except Exception as e:
                print(f"  Error processing T multiplier {multiplier}: {e}")

# 主程序
if __name__ == "__main__":
    # 运行增强的f_pbh分析
    create_enhanced_fpbh_analysis(model_name='model_3')