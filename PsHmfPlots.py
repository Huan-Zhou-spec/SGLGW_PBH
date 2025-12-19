""""
Script for plotting the power spectrum, variance and halo mass function in PBHs scenarios
Author: Pablo Villanueva Domingo
Started: October 2018
Last update: March 2021
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from modules import halo_mass_function, IsoPS, PkPBH
from modules import sis_velocity_dispersion
from colossus.cosmology import cosmology




cosmo = cosmology.setCosmology('planck18')
# 设置字体以避免警告 - 使用更通用的字体
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

class PlotConfig:
    """Graphic configuration class"""
    
    def __init__(self):
        # 图形参数
        self.figsize = (24, 6)
        self.dpi = 100
        self.facecolor = 'white'
        
        # 通用字体大小
        self.font_sizes = {
            'title': 13,
            'axis_label': 14,
            'tick_label': 16,
        }
        
        # 功率谱图参数
        self.pow_xlim = (1e-4, 1e1)
        self.pow_ylim = (1e-2, 1e5)
        self.pow_xlabel = r'$k$ [$h$/Mpc]'
        self.pow_ylabel = r'$P(k)$ [$({\rm Mpc}/h)^3$]'
        self.pow_line_width = 2.0
        self.pow_grid_alpha = 0.3
        self.pow_legend_loc = 3
        self.pow_legend = 18
        
        
        # 质量函数图参数（质量空间）
        self.halo_ylim_m = (1e-7, 1e1)
        self.halo_xlim_m = (1e8, 1e13)
        self.halo_xlabel_m = r'$M_{\rm h}$ [$M_{\odot}/h$]'
        self.halo_ylabel_m = r'd$n$/dln$M_{\rm h}$ [$(h$/Mpc$)^3$]'
        self.halo_line_width_m = 2.0
        self.halo_grid_alpha_m = 0.3
        self.halo_legend_loc_m = 3
        self.halo_legend = 18
    
        
        # 质量函数图参数（速度空间）
        self.halo_ylim_v = (1e-6, 1e0)
        self.halo_xlim_v = (0, 400)  # 合理的速度分散范围
        self.halo_xlabel_v = r'$\sigma_{\rm sis}$ [km/s]'
        self.halo_ylabel_v = r'd$n$/d$\sigma_{\rm sis}$ [$(h$/Mpc$)^3$/(km/s)]'
        self.halo_line_width_v = 2.0
        self.halo_grid_alpha_v = 0.3
        self.halo_legend_loc_v = 1

        
        # 保存参数
        self.save_filename = "Plots/combined_power_spectrum_halo_mf.pdf"
        self.save_dpi = 300
        self.save_bbox_inches = 'tight'
        self.save_facecolor = 'white'

def setup_axes_style(ax, config, space_type='pow'):
    """Set the coordinate axis styles uniformly"""
    font_sizes = config.font_sizes
    
    # 设置标签字体大小
    ax.set_xlabel(ax.get_xlabel(), fontsize=font_sizes['axis_label'])
    ax.set_ylabel(ax.get_ylabel(), fontsize=font_sizes['axis_label'])
    ax.set_title(ax.get_title(), fontsize=font_sizes['title'])
    
    # 设置刻度标签大小
    ax.tick_params(axis='both', which='major', 
                  labelsize=font_sizes['tick_label'],
                  length=6, width=1.5)
    ax.tick_params(axis='both', which='minor', 
                  labelsize=font_sizes['tick_label']-1,
                  length=4, width=1)
    
    # 设置网格
    if space_type == 'pow':
        grid_alpha = config.pow_grid_alpha
    elif space_type == 'mass':
        grid_alpha = config.halo_grid_alpha_m
    else:  # velocity
        grid_alpha = config.halo_grid_alpha_v
        
    ax.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.8)
    

def plot_power_spectrum(ax, k, fpbh, MassPBH, xcl, xi0, config):
    """Draw the power spectrum"""
    # Draw different models
    ax.loglog(k, cosmo.matterPowerSpectrum(k), 'k-', 
              label=r"$\Lambda$CDM", linewidth=config.pow_line_width)
    
    ax.loglog(k, IsoPS(k, fpbh, MassPBH, xcl, xi0, 'clu'), 
             color='red', linestyle="-", 
             label=r"PBH (Cluster)", 
             linewidth=config.pow_line_width)
    ax.loglog(k, IsoPS(k, fpbh, MassPBH, xcl, xi0, 'pos'), 
             color='red', linestyle="--", 
             label=r"PBH (Poisson)", 
             linewidth=config.pow_line_width)
    ax.loglog(k, PkPBH(k, fpbh, MassPBH, xcl, xi0, 'clu'), 
             color='blue', linestyle="-", 
             label=r"$\Lambda$CDM+PBH (Cluster)", 
             linewidth=config.pow_line_width)
    ax.loglog(k, PkPBH(k, fpbh, MassPBH, xcl, xi0, 'pos'), 
             color='green', linestyle="--", 
             label=r"$\Lambda$CDM+PBH (Poisson)", 
             linewidth=config.pow_line_width)
    
    # 设置坐标轴标签和范围
    ax.set_xlabel(config.pow_xlabel)
    ax.set_ylabel(config.pow_ylabel)
    ax.set_xlim(config.pow_xlim)
    ax.set_ylim(config.pow_ylim)
    #ax.set_title("Power Spectrum")
    
    # 添加图例
    ax.legend(fontsize=config.pow_legend, 
              loc=config.pow_legend_loc, framealpha=0.9)
    
    # 设置坐标轴样式
    setup_axes_style(ax, config, 'pow')

def plot_mass_function(ax, M, z, fpbh, MassPBH, xcl, xi0, config, space='mass'):
    """绘制质量函数（质量空间或速度空间）"""
    # 定义模型配置
    models = [
        ('CDM', 'st', 'k-', r"$\Lambda$CDM", 2.0)
        #('CDM', 'st', 'k-', r"CDM (ST)", 2.0),
        #('CDM', 'jk', 'k--', r"CDM (Jenkins)", 2.0)
    ]
    
    pbh_models = [
        ('PBH', 'st', 'clu', 'blue', '-', "Cluster", 2.0),
        ('PBH', 'st', 'pos', 'green', '-', "Poisson", 2.0)
        #('PBH', 'st', 'clu', 'blue', '-', "ST, Cluster", 2.0),
        #('PBH', 'st', 'pos', 'green', '-', "ST, Poisson", 2.0),
        #('PBH', 'jk', 'clu', 'blue', '--', "Jenkins, Cluster", 2),
        #('PBH', 'jk', 'pos', 'green', '--', "Jenkins, Poisson", 2)
    ]
    
    # 绘制CDM模型
    for model_type, model, linestyle, label, lw in models:
        dndlnM = halo_mass_function(M, z, model_type, model=model)
        if space == 'velocity':
            sigma = sis_velocity_dispersion(z, M, None) #km/s
            dM_dsigma = np.gradient(M, sigma)
            y_data = dndlnM / M * dM_dsigma
            x_data = sigma
            line_width = config.halo_line_width_v
            ax.semilogy(x_data, y_data, linestyle, label=label, 
                     linewidth=lw, alpha=0.9)
        else:
            y_data = dndlnM
            x_data = M
            line_width = config.halo_line_width_m
            ax.loglog(x_data, y_data, linestyle, label=label, 
                 linewidth=lw, alpha=0.9)
    
    # 绘制PBH模型
    for model_type, model, mode, color, linestyle, desc, lw in pbh_models:
        dndlnM = halo_mass_function(M, z, model_type, model=model,
                                  fpbh=fpbh, Mpbh=MassPBH, xcl=xcl, xi0=xi0, mode=mode)
        label = rf"$\Lambda$CDM+PBH ({desc})"
        if space == 'velocity':
            sigma = sis_velocity_dispersion(z, M, None)
            dM_dsigma = np.gradient(M, sigma)
            y_data = dndlnM / M * dM_dsigma
            #y_data = dndlnM *3/sigma
            x_data = sigma
            line_width = config.halo_line_width_v
            ax.semilogy(x_data, y_data, color=color, linestyle=linestyle, 
                     label=label, linewidth=lw, alpha=0.9)
        else:
            y_data = dndlnM
            x_data = M
            line_width = config.halo_line_width_m
            ax.loglog(x_data, y_data, color=color, linestyle=linestyle, 
                 label=label, linewidth=lw, alpha=0.9)
    
    # 设置坐标轴标签和范围
    if space == 'velocity':
        ax.set_xlabel(config.halo_xlabel_v)
        ax.set_ylabel(config.halo_ylabel_v)
        ax.set_ylim(config.halo_ylim_v)
        ax.set_xlim(config.halo_xlim_v)
        #ax.set_title("Halo Mass Function (Velocity Space)")
        legend_loc = config.halo_legend_loc_v
        space_type = 'velocity'
    else:
        ax.set_xlabel(config.halo_xlabel_m)
        ax.set_ylabel(config.halo_ylabel_m)
        ax.set_ylim(config.halo_ylim_m)
        ax.set_xlim(config.halo_xlim_m)
        #ax.set_title("Halo Mass Function (Mass Space)")
        legend_loc = config.halo_legend_loc_m
        space_type = 'mass'
    
    # 添加图例
    ax.legend(fontsize=config.halo_legend, 
              loc=legend_loc, framealpha=0.9)
    
    # 设置坐标轴样式
    setup_axes_style(ax, config, space_type)


#--- 主程序 ---#
# 设置参数
MassPBH = 1e9
fracs = 10**-3
xcl = 1 # Mpc
xi0 = 10
z = 7

# 定义计算范围
k = np.logspace(-4, 1, num=100)  # h/Mpc
M = np.logspace(8, 15, num=100)  # M_sun/h

# 创建配置实例
config = PlotConfig()

# 创建图形和子图
fig, (ax_pow, ax_halo_m, ax_halo_v) = plt.subplots(1, 3, 
                                                   figsize=config.figsize, 
                                                   dpi=config.dpi, 
                                                   facecolor=config.facecolor)


# 绘制功率谱
plot_power_spectrum(ax_pow, k, fracs, MassPBH, xcl, xi0, config)

# 绘制质量空间的质量函数
plot_mass_function(ax_halo_m, M, z, fracs, MassPBH, xcl, xi0, config, space='mass')

# 绘制速度空间的质量函数
plot_mass_function(ax_halo_v, M, z, fracs, MassPBH, xcl, xi0, config, space='velocity')

# 调整子图间距
plt.subplots_adjust(wspace=0.25, top=0.92)

# 保存图形
plt.savefig(config.save_filename, 
           dpi=config.save_dpi, 
           bbox_inches=config.save_bbox_inches, 
           facecolor=config.save_facecolor)

print(f"图形已保存至: {config.save_filename}")

# 显示图形
plt.show()