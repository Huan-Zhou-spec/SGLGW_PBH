#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 14:06:46 2025

@author: ubuntu
"""

import numpy as np
import matplotlib.pyplot as plt
#from Lensis import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import time

class DeltaFunctionIntegral:
    """
    Class for computing ∫∫∫ g(x,y,z) δ(a - x⁴K(y,z)) dxdydz
    Vectorized version for improved performance
    """
    
    def __init__(self, a, K_func, K_deriv_y, K_deriv_z):
        """
        Initialize parameters
        a: constant in delta function
        K_func: function K(y,z) - should accept vector inputs
        K_deriv_y: function for ∂K/∂y - should accept vector inputs  
        K_deriv_z: function for ∂K/∂z - should accept vector inputs
        """
        self.a = a
        self.K_func = K_func
        self.K_deriv_y = K_deriv_y
        self.K_deriv_z = K_deriv_z
    
    def x_on_surface(self, y, z):
        """Calculate x coordinate on the surface f=0 (vectorized)"""
        # 首先检查 y < z 条件
        valid_mask = y < z
        
        # 初始化结果数组
        x_vals = np.full_like(y, np.nan, dtype=float)
        
        # 只在有效区域计算
        K_val = self.K_func(y[valid_mask], z[valid_mask])
        positive_K_mask = K_val > 0
        
        # 创建有效计算掩码
        calc_mask = valid_mask.copy()
        calc_mask[calc_mask] = positive_K_mask
        
        # 计算 x 值
        x_vals[calc_mask] = (self.a / K_val[positive_K_mask]) ** 0.25
        
        return x_vals
    
    def gradient_norm(self, y, z):
        """Calculate |∇f| on the surface (vectorized)"""
        # 获取 x 值
        x_vals = self.x_on_surface(y, z)
        
        # 初始化结果数组
        grad_norms = np.full_like(y, np.nan, dtype=float)
        
        # 找到有效点 (x 值存在且有效)
        valid_mask = ~np.isnan(x_vals)
        
        if not np.any(valid_mask):
            return grad_norms
        
        # 计算梯度范数
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        x_valid = x_vals[valid_mask]
        
        K_val = self.K_func(y_valid, z_valid)
        #dKdy = self.K_deriv_y(y_valid, z_valid)
        #dKdz = self.K_deriv_z(y_valid, z_valid)
        
        term1 = 16 * K_val**2
        #term2 = x_valid**2 * (dKdy**2 + dKdz**2)
        
        grad_norms[valid_mask] = abs(x_valid**3 * np.sqrt(term1))
        
        return grad_norms
    
    def create_integrand(self, g_func):
        """Create vectorized integrand function"""
        def integrand(y, z):
            # 确保输入为数组
            y_arr = np.asarray(y)
            z_arr = np.asarray(z)
            
            # 初始化结果数组
            result = np.zeros_like(y_arr, dtype=float)
            
            # 检查 y < z 条件
            valid_mask = y_arr < z_arr
            
            if not np.any(valid_mask):
                return result
            
            # 计算 x 值和梯度范数
            x_vals = self.x_on_surface(y_arr[valid_mask], z_arr[valid_mask])
            grad_norms = self.gradient_norm(y_arr[valid_mask], z_arr[valid_mask])
            
            # 找到可计算点
            calc_mask = valid_mask.copy()
            calc_mask[calc_mask] = (~np.isnan(x_vals)) & (~np.isnan(grad_norms)) & (grad_norms > 0)
            
            if not np.any(calc_mask):
                return result
            
            # 计算被积函数
            y_calc = y_arr[calc_mask]
            z_calc = z_arr[calc_mask]
            x_calc = x_vals[calc_mask[valid_mask]]
            grad_calc = grad_norms[calc_mask[valid_mask]]
            
            result[calc_mask] = g_func(x_calc, y_calc, z_calc) / grad_calc
            
            return result
        return integrand
    
    
    def compute_integral_vectorized(self, g_func, y_range, z_range, n_points=100):
        """
        Vectorized integration method for improved performance
        """
        integrand = self.create_integrand(g_func)
        
        y_min, y_max = y_range
        z_min, z_max = z_range
        
        # 创建网格
        y_vals = np.linspace(y_min, y_max, n_points)
        z_vals = np.linspace(z_min, z_max, n_points)
        Y, Z = np.meshgrid(y_vals, z_vals, indexing='ij')
        
        # 向量化计算被积函数
        integrand_vals = integrand(Y, Z)
        
        # 使用梯形法则进行数值积分
        dy = (y_max - y_min) / (n_points - 1)
        dz = (z_max - z_min) / (n_points - 1)
        
        result = np.trapz(np.trapz(integrand_vals, dx=dz, axis=1), dx=dy)
        
        return result
    
    def compute_integral_mcmc(self, g_func, y_range, z_range, n_points=10000, return_error=False):
        """
        蒙特卡洛积分方法（带误差估计）
        """
        integrand = self.create_integrand(g_func)
    
        y_min, y_max = y_range
        z_min, z_max = z_range
    
        # 计算积分区域面积
        area = (y_max - y_min) * (z_max - z_min)
    
        # 在积分区域内随机采样
        y_rand = np.random.uniform(y_min, y_max, n_points)
        z_rand = np.random.uniform(z_min, z_max, n_points)
    
        # 计算被积函数在随机点的值
        integrand_vals = integrand(y_rand, z_rand)
    
        # 蒙特卡洛积分估计
        mean_val = np.mean(integrand_vals)
        result = area * mean_val
    
        if return_error:
            # 计算标准误差
            std_val = np.std(integrand_vals, ddof=1)
            error = area * std_val / np.sqrt(n_points)
            return result, error
        else:
            return result
    
    
    def compute_integral_triangle_vectorized(self, g_func, y_range, z_range, n_points=100):
        """
        Vectorized version for triangular integration domain
        """
        integrand = self.create_integrand(g_func)
        
        y_min, y_max = y_range
        z_min, z_max = z_range
        
        # 创建 y 坐标
        y_vals = np.linspace(y_min, y_max, n_points)
        
        total_integral = 0.0
        
        for i in range(len(y_vals) - 1):
            y_current = y_vals[i]
            y_next = y_vals[i + 1]
            
            # 对于每个 y 区间，z 的范围是 [max(y_next, z_min), z_max]
            z_low = max(y_next, z_min)  # 确保 z > y
            z_high = z_max
            
            if z_low < z_high:
                # 在 z 方向创建向量
                z_vals_segment = np.linspace(z_low, z_high, n_points)
                
                # 创建 y 的常数数组（使用 y_next 作为当前段的代表值）
                y_constant = np.full_like(z_vals_segment, y_next)
                
                # 向量化计算 z 方向的积分
                z_integrand = integrand(y_constant, z_vals_segment)
                z_integral = np.trapz(z_integrand, z_vals_segment)
                
                # 在 y 方向累加
                total_integral += z_integral * (y_next - y_current)
        
        return total_integral
    
    def compute_integral_adaptive(self, g_func, y_range, z_range, initial_n_points=50, tolerance=1e-6, max_refinements=5):
        """
        Adaptive integration method with error estimation
        """
        # 初始计算
        result_prev = self.compute_integral_vectorized(g_func, y_range, z_range, initial_n_points)
        
        for refinement in range(max_refinements):
            # 增加点数
            n_points = initial_n_points * (2 ** (refinement + 1))
            result_current = self.compute_integral_vectorized(g_func, y_range, z_range, n_points)
            
            # 检查收敛
            error = abs(result_current - result_prev) / abs(result_current)
            if error < tolerance:
                return result_current, error, n_points
            
            result_prev = result_current
        
        # 如果未达到容差，返回最后一次计算的结果
        return result_current, error, n_points
    
    def get_integration_domain(self, y_range, z_range, n_points=50):
        """
        获取有效的积分区域（满足 y < z 的区域）
        """
        y_min, y_max = y_range
        z_min, z_max = z_range
        
        # 实际的有效区域
        actual_y_min = y_min
        actual_y_max = min(y_max, z_max)  # y 不能超过 z_max
        actual_z_min = max(z_min, y_min)  # z 不能小于 y_min
        actual_z_max = z_max
        
        return (actual_y_min, actual_y_max), (actual_z_min, actual_z_max)
    
    def visualize_integrand(self, g_func, y_range, z_range, n_points=100, filename=None):
        """
        可视化被积函数，用于调试和验证
        """
        integrand = self.create_integrand(g_func)
        
        y_min, y_max = y_range
        z_min, z_max = z_range
        
        # 创建网格
        y_vals = np.linspace(y_min, y_max, n_points)
        z_vals = np.linspace(z_min, z_max, n_points)
        Y, Z = np.meshgrid(y_vals, z_vals, indexing='ij')
        
        # 计算被积函数
        integrand_vals = integrand(Y, Z)
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制被积函数
        plt.imshow(integrand_vals.T, extent=[y_min, y_max, z_min, z_max], 
                  origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label='Integrand Value')
        
        # 添加对角线 (y=z)
        plt.plot([y_min, min(y_max, z_max)], [y_min, min(y_max, z_max)], 'r--', label='y=z')
        
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title('Integrand Visualization')
        plt.legend()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return integrand_vals

'''
# 定义示例函数
# 创建两个分析实例
start_time = time.time()
analysis_cdm = LensingAnalysis(cosmo, cosmology_type='CDM')
analysis_pbh_p = LensingAnalysis(cosmo, cosmology_type='PBH', mode='pos')  # 假设PBH是有效的cosmology_type
analysis_pbh_c = LensingAnalysis(cosmo, cosmology_type='PBH', mode='clu') 

# 获取两个模型的核函数
K_func_cdm = analysis_cdm.kernel.K
K_deriv_y_cdm = analysis_cdm.kernel.K_zl
K_deriv_z_cdm = analysis_cdm.kernel.K_zs

g_func_cdm = analysis_cdm.g
g_func_pbhp = analysis_pbh_p.g
g_func_pbhc = analysis_pbh_c.g


# 测试多个 a 值
a_values = np.logspace(0, 5, 50)

# 积分区域
y_range = (1e-3, 7.0)
z_range = (1e-3, 7.0)

# 存储两个模型的结果
results_cdm = []
results_pbh_p = []
results_pbh_c = []
print("计算多个 a 值的积分...")
for i, a in enumerate(a_values):
    print(f"计算 a = {a:.2e}...")
    
    # 创建两个积分器实例
    integrator = DeltaFunctionIntegral(a, K_func_cdm, K_deriv_y_cdm, K_deriv_z_cdm)
    
    # 计算两个模型的积分
    result_cdm = integrator.compute_integral_vectorized(g_func_cdm, y_range, z_range, n_points=100)
    result_pbh_p = integrator.compute_integral_vectorized(g_func_pbhp, y_range, z_range, n_points=100)
    result_pbh_c = integrator.compute_integral_vectorized(g_func_pbhc, y_range, z_range, n_points=100)
    
    results_cdm.append(result_cdm)
    results_pbh_p.append(result_pbh_p)
    results_pbh_c.append(result_pbh_c)

# 转换为 numpy 数组
results_cdm = np.array(results_cdm)
results_pbh_p = np.array(results_pbh_p)
results_pbh_c = np.array(results_pbh_c)


plt.figure(figsize=(10, 8))

# 在同一个图中绘制两个模型的结果
plt.loglog(a_values, results_cdm, '-', linewidth=2, markersize=4, color='green', label='CDM')
plt.loglog(a_values, results_pbh_p, '-', linewidth=2, markersize=4, color='red', label='PBH+CDM (Poisson)')
plt.loglog(a_values, results_pbh_c, '-', linewidth=2, markersize=4, color='b', label='PBH+CDM (Cluster)')
plt.xlabel(r'$\Delta t$ (hrs)', fontsize=18)
plt.ylabel(r'$p(\Delta t|\Omega)$', fontsize=18)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=21)

plt.tight_layout()
plt.savefig('Plots/nt_comparison.pdf')
plt.show()

# 打印总结信息
print(f"\n总结:")
print(f"a 值范围: {a_values[0]:.2e} 到 {a_values[-1]:.2e}")
print(f"积分区域: y ∈ {y_range}, z ∈ {z_range}")
end_time = time.time()
# 计算并打印运行时间
elapsed_time = (end_time - start_time)/60
print(f"代码运行了 {elapsed_time} mins")
'''


'''
# 创建包含两个子图的图表
fig, (ax1, ax2, ax3) = plt.subplots(1, 1, figsize=(9, 6))

# 第一个子图：三个模型的结果对比
ax1.semilogx(a_values, results_cdm, '-', linewidth=2, markersize=4, color='green', label='CDM')
ax1.semilogx(a_values, results_pbh_p, '--', linewidth=2, markersize=4, color='red', label='PBH+CDM (Poisson)')
ax1.semilogx(a_values, results_pbh_c, '.', linewidth=2, markersize=4, color='b', label='PBH+CDM (Cluster)')
ax1.set_xlabel(r'$\Delta t$ (hrs)', fontsize=18)
ax1.set_ylabel(r'$p(\Delta t|\Omega)$', fontsize=18)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=18)

# 第二个子图：CDM模型的被积函数
selected_a = a_values[len(a_values)//2]
integrator_cdm = DeltaFunctionIntegral(selected_a, K_func_cdm, K_deriv_y_cdm, K_deriv_z_cdm)

n_vis = 100
y_vals = np.linspace(y_range[0], y_range[1], n_vis)
z_vals = np.linspace(z_range[0], z_range[1], n_vis)
Y, Z = np.meshgrid(y_vals, z_vals, indexing='ij')
integrand_cdm = integrator_cdm.create_integrand(g_func_cdm)
integrand_map_cdm = integrand_cdm(Y, Z)

im1 = ax2.imshow(integrand_map_cdm.T, 
                extent=[y_range[0], y_range[1], z_range[0], z_range[1]], 
                origin='lower', aspect='auto', cmap='viridis')
ax2.set_xlabel(r'$z_{\rm l}$', fontsize=16)
ax2.set_ylabel(r'$z_{\rm s}$', fontsize=16)
ax2.set_title(f'CDM: $\Delta t$={selected_a:.1f} hrs')
ax2.plot([y_range[0], min(y_range[1], z_range[1])], 
        [y_range[0], min(y_range[1], z_range[1])], 
        'r-', linewidth=1, label=r'$z_{\rm l}=z_{\rm s}$')
ax2.legend(loc=4, fontsize=12)
plt.colorbar(im1, ax=ax2, shrink=0.8)

# 第三个子图：PBH模型的被积函数
integrator_pbh = DeltaFunctionIntegral(selected_a, K_func_pbh, K_deriv_y_pbh, K_deriv_z_pbh)
integrand_pbh = integrator_pbh.create_integrand(g_func_pbh)
integrand_map_pbh = integrand_pbh(Y, Z)

im2 = ax3.imshow(integrand_map_pbh.T, 
                extent=[y_range[0], y_range[1], z_range[0], z_range[1]], 
                origin='lower', aspect='auto', cmap='viridis')
ax3.set_xlabel(r'$z_{\rm l}$', fontsize=16)
ax3.set_ylabel(r'$z_{\rm s}$', fontsize=16)
ax3.set_title(f'PBH: $\Delta t$={selected_a:.1f} hrs')
ax3.plot([y_range[0], min(y_range[1], z_range[1])], 
        [y_range[0], min(y_range[1], z_range[1])], 
        'r-', linewidth=1, label=r'$z_{\rm l}=z_{\rm s}$')
ax3.legend(loc=4, fontsize=12)
plt.colorbar(im2, ax=ax3, shrink=0.8)

plt.tight_layout()
plt.savefig('Plots/nt_comparison_detailed.pdf')
plt.show()
'''





