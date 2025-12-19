#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:03:49 2021

@author: zhouhuan
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d

# 设置全局字体和图形参数
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 18

def load_data(filename):
    """加载数据文件"""
    return np.loadtxt(filename, unpack=True)

def create_fill_data(x_data, y_max=1.0):
    """创建填充数据"""
    return [y_max] * len(x_data)

def setup_plot():
    """设置绘图参数"""
    fig, ax = plt.subplots(figsize=(14, 10))
    return fig, ax

def plot_constraint(ax, x_data, y_data, color, label, text_pos, rotation=0, alpha=0.2):
    """绘制单个约束曲线和填充区域"""
    # 绘制约束线
    ax.loglog(x_data, y_data, color=color, linestyle='-', linewidth=2.5, label=label)
    
    # 填充区域
    y_fill = create_fill_data(x_data)
    ax.fill_between(x_data, y_data, y_fill, facecolor=color, alpha=alpha)
    
    # 添加文本标签
    ax.text(text_pos[0], text_pos[1], label, fontsize=24, 
            ha='center', va='bottom', rotation=rotation, color=color)

def main():
    """主函数"""
    # 加载所有约束数据
    data_files = {
        'LSS': "data/fpbh_bound/LSS.txt",
        'Dynamical': "data/fpbh_bound/Dynamical.txt", 
        'Accretion': "data/fpbh_bound/Accretion.txt",
        'Evaporation': "data/fpbh_bound/Evaporation.txt",
        'GWs': "data/fpbh_bound/GWs.txt",
        'Microlensing': "data/fpbh_bound/Microlensing.txt"
    }
    
    # 存储所有数据
    constraints = {}
    for name, filename in data_files.items():
        m, f = load_data(filename)
        constraints[name] = {
            'mass': 10**m if name == 'LSS' else m,
            'f_pbh': 10**f if name == 'LSS' else f,
            'color': None,
            'text_pos': None,
            'rotation': 0
        }
    
    # 设置约束参数
    constraint_params = {
        'LSS': {'color': 'g', 'text_pos': (1e12, 9e-4), 'rotation': 75},
        'Dynamical': {'color': 'c', 'text_pos': (9e3, 2e-3), 'rotation': -75},
        'Accretion': {'color': 'olive', 'text_pos': (1e1, 8e-5), 'rotation': 90},
        'Evaporation': {'color': 'purple', 'text_pos': (1e-16, 7e-3), 'rotation': 85},
        'GWs': {'color': 'orange', 'text_pos': (1e-1, 7e-3), 'rotation': -60},
        'Microlensing': {'color': 'r', 'text_pos': (9e-8, 2e-3), 'rotation': 60}
    }
    
    # 理论曲线数据
    mb = np.array([1e6, 1e7, 1e8, 1e9, 1e10])
    f1 = np.array([10**(-2.5998), 10**(-3.0596), 10**(-3.5813), 10**(-3.7578), 10**(-3.7228)]) #model_2, R=5e5yr^-1, Tobs=10yrs
    f2 = np.array([10**(-2.4618), 10**(-3.0526), 10**(-3.2647), 10**(-3.4653), 10**(-3.5778)]) #model_2, R=1e5yr^-1, Tobs=10yrs
    #f3 = np.array([10**(-4.0934), 10**(-4.1232), 10**(-4.1094), 10**(-4.0200), 10**(-3.8774)]) #model_3, R=5e5yr^-1, Tobs=10yrs, xcl=1
    #f4 = np.array([10**(-3.9613), 10**(-3.9508), 10**(-3.9456), 10**(-3.8866), 10**(-3.7594)]) #model_3, R=1e5yr^-1, Tobs=10yrs, xcl=1
    f5 = np.array([10**(-4.0630), 10**(-4.0637), 10**(-4.0662), 10**(-4.0655), 10**(-4.0496)]) #model_3, R=5e5yr^-1, Tobs=10yrs, xcl=10
    f6 = np.array([10**(-3.9448), 10**(-3.9471), 10**(-3.9438), 10**(-3.9381), 10**(-3.9346)]) #model_3, R=1e5yr^-1, Tobs=10yrs, xcl=10
    
    f1_interpolator = interp1d(
        np.log10(mb),
        np.log10(f1),
        kind='cubic',  # 线性插值，也可以选择 'linear' 或 'quadratic'
        bounds_error=False,
        fill_value='extrapolate'  # 允许外推
    )
    
    f5_interpolator = interp1d(
        np.log10(mb),
        np.log10(f5),
        kind='cubic',  # 线性插值，也可以选择 'linear' 或 'quadratic'
        bounds_error=False,
        fill_value='extrapolate'  # 允许外推
    )
    
    m_new= np.logspace(np.log10(mb.min()), np.log10(mb.max()), 100)
    
    # 创建图形
    fig, ax = setup_plot()
    
    # 绘制理论曲线
    ax.loglog(m_new, 10**f1_interpolator(np.log10(m_new)), 'k-', linewidth=3.5, 
              label=r'$\Lambda$CDM+PBH Poisson')
    ax.loglog(m_new, 10**f5_interpolator(np.log10(m_new)), 'k--', linewidth=3.5, 
              label=r'$\Lambda$CDM+PBH Cluster ')
    f_fill = create_fill_data(m_new)
    ax.fill_between(m_new, 10**f1_interpolator(np.log10(m_new)), f_fill, facecolor='k', alpha=0.2)
    ax.fill_between(m_new, 10**f5_interpolator(np.log10(m_new)), f_fill, facecolor='k', alpha=0.2)
    
    # 绘制所有约束
    for name, params in constraint_params.items():
        data = constraints[name]
        plot_constraint(
            ax, data['mass'], data['f_pbh'], 
            params['color'], name, 
            params['text_pos'], params['rotation']
        )
    
    # 设置坐标轴和标签
    ax.set_xlabel(r'$M_{\rm PBH}~(M_{\odot})$', labelpad=10, fontsize=24)
    ax.set_ylabel(r'$f_{\rm PBH}$', labelpad=10, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    
    # 设置坐标范围
    ax.set_xlim(1e-18, 1e14)
    ax.set_ylim(1e-6, 1)
    
    # 添加图例
    ax.legend(fontsize=21, loc='lower left', framealpha=0.9)
    
    # 添加网格
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # 调整布局并显示
    plt.tight_layout()
    
    # 保存图形
    plt.savefig("Plots/fpbh.pdf", dpi=300, bbox_inches='tight')
    print("图形已保存为 fpbh.pdf")
    plt.show()
    

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"程序运行时间: {end_time - start_time:.2f} 秒")






