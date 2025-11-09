#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 15:37:33 2025

@author: ubuntu
"""

from modules import DeltaFunctionIntegral
from modules import LensingAnalysis
import numpy as np
import matplotlib.pyplot as plt
import time


start_time = time.time()
# 定义宇宙学模型列表
cosmology_types = ['CDM', 'CDM+PBH', 'CDM+PBH']
modes = ['pos', 'pos', 'clu']
models = [1,2,3]
colors = ['g', 'r', 'b']  # CDM用蓝色，PBH用红色和蓝色
all_results = {}

# 测试多个 a 值
a_values = np.logspace(0, 6, 60)

# 积分区域
y_range = (1e-3, 7.0)
z_range = (1e-3, 7.0)

print("计算多个a值的积分...")
for cosmo_type, mode, color, model in zip(cosmology_types, modes, colors, models):
    print(f"\n=== 计算 {cosmo_type} 模型 ===")
    
    # 初始化对应宇宙学模型的分析
    analysis = LensingAnalysis(cosmology_type=cosmo_type, mode=mode)
    K_func = analysis.kernel.K
    K_deriv_y = analysis.kernel.K_zl
    K_deriv_z = analysis.kernel.K_zs
    g_func = analysis.g
    
    results = []
    
    for i, a in enumerate(a_values):
        if i % 10 == 0:  # 每10个点打印一次进度
            print(f"计算 a = {a:.2e}...")
        
        # 创建积分器实例
        integrator = DeltaFunctionIntegral(a, K_func, K_deriv_y, K_deriv_z)
        
        # 计算积分
        result = integrator.compute_integral_vectorized(g_func, y_range, z_range, n_points=100)
        results.append(result)
    
    # 存储结果
    all_results[model] = np.array(results)
    
    # 保存数据
    data_to_save = np.column_stack((a_values, all_results[model]))
    np.savetxt(f'data/nt_data_{cosmo_type}_{mode}.txt', data_to_save, 
               header='Delta_t(hrs) p(Delta_t|Omega)', fmt='%.6e')
    print(f"{cosmo_type} 数据已保存到 data/nt_data_{cosmo_type}_{mode}.txt")

# 创建对比图表
plt.figure(figsize=(10, 8))
modes_label = ['', 'Posison', 'Cluster']
# 1. 积分结果 vs a 值对比（上子图）
for cosmo_type, mode, color, model in zip(cosmology_types, modes_label, colors, models):
    plt.semilogx(a_values, all_results[model], '-', linewidth=2, 
                 markersize=4, color=color, label=f'{cosmo_type} {mode}')

plt.xlabel(r'$\Delta t$ (hrs)', fontsize=18)
plt.ylabel(r'$p(\Delta t|\Omega)$', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(a_values[0], a_values[-1])
plt.legend(fontsize=21)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Plots/nt_CDM_vs_PBH.pdf', dpi=300, bbox_inches='tight')
plt.show()
end_time = time.time()
# 计算并打印运行时间
elapsed_time = (end_time - start_time)/60
print(f"代码运行了 {elapsed_time} mins")

