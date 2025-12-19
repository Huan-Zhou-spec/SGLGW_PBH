import numpy as np
import h5py
import os
from scipy.interpolate import RegularGridInterpolator, interp1d
from modules import read_hdf5_data  # 使用您现有的读取函数


def process_hdf5_to_p_delta_t(input_h5_path, output_h5_path, n_delta_t_points=200):
    """
    将原始HDF5文件转换为p(Δt)分布版本
    保持与原始文件相同的结构，以便read_hdf5_data能正常读取
    """
    print(f"处理文件: {input_h5_path}")
    
    # 1. 使用现有函数读取原始数据
    data = read_hdf5_data(input_h5_path)
    
    # 2. 提取数据
    f_pbh_values = data['f_pbh_values']
    a_values = data['a_values']
    
    # 3. 创建新的Δt网格
    a_min, a_max = a_values.min(), a_values.max()
    delta_t_min = a_min
    delta_t_max = a_max
    delta_t_grid = np.logspace(np.log10(delta_t_min), np.log10(delta_t_max), n_delta_t_points)
    
    print(f"原始网格: {len(a_values)}点, 范围: [{a_min:.2e}, {a_max:.2e}]")
    print(f"新Δt网格: {len(delta_t_grid)}点, 范围: [{delta_t_min:.2e}, {delta_t_max:.2e}]")
    
    # 4. 处理每个模型
    models_data_new = {}
    
    for model_name, model_data in data['models_data'].items():
        print(f"  处理模型: {model_name}")
        
        integral_results_original = model_data['integral_results']
        N_lens_original = model_data['N_lens']
        cosmology_type = model_data['cosmology_type']
        mode = model_data['mode']
        
        # 创建原始数据插值器
        interp = RegularGridInterpolator(
            (f_pbh_values, a_values), 
            integral_results_original,
            method='cubic',
            bounds_error=False,
            fill_value=0.0
        )
        
        # 计算新的p(Δt)分布
        n_fpbh = len(f_pbh_values)
        n_delta_t = len(delta_t_grid)
        integral_results_new = np.zeros((n_fpbh, n_delta_t))
        
        for i, f_pbh in enumerate(f_pbh_values):
            # 显示进度
            if i % 10 == 0:
                print(f"    进度: {i+1}/{n_fpbh} (f_pbh={f_pbh:.3e})")
            
            # 在原始网格上计算p̃(u)
            p_tilde_u = np.array([interp((f_pbh, u)) for u in a_values])
            
            # 计算g(u) = p̃(u)/u²
            u_safe = np.where(a_values > 0, a_values, 1e-10)
            g_u = p_tilde_u / (u_safe**2)
            
            # 计算累积积分G(u) = ∫_u^∞ g(u') du'
            G_u = np.zeros_like(a_values)
            for j in range(len(a_values)-2, -1, -1):
                du = a_values[j+1] - a_values[j]
                trap_area = 0.5 * (g_u[j] + g_u[j+1]) * du
                G_u[j] = G_u[j+1] + trap_area
            
            # 创建G(u)插值器
            G_interp = interp1d(a_values, G_u, kind='cubic', 
                               bounds_error=False, fill_value=(G_u[0], 0.0))
            
            # 计算p(Δt) = 2Δt * G(Δt)
            for k, dt in enumerate(delta_t_grid):
                if dt <= 0 or dt >= a_max:
                    integral_results_new[i, k] = 0.0
                else:
                    G_val = G_interp(dt)
                    integral_results_new[i, k] = 2 * dt * G_val
            
            # 归一化
            integral_val = np.trapz(integral_results_new[i, :], delta_t_grid)
            if integral_val > 0:
                integral_results_new[i, :] = integral_results_new[i, :] / integral_val
        
        # 保存处理后的模型数据
        models_data_new[model_name] = {
            'integral_results': integral_results_new,
            'N_lens': N_lens_original,
            'cosmology_type': cosmology_type,
            'mode': mode
        }
    
    # 5. 保存到新的HDF5文件（保持与原始文件相同的结构）
    print(f"\n保存到: {output_h5_path}")
    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
    
    with h5py.File(output_h5_path, 'w') as f:
        # 保存原始属性
        f.attrs['R'] = data['R']
        f.attrs['T_obs'] = data['T_obs']
        f.attrs['y_range_min'] = data['y_range'][0]
        f.attrs['y_range_max'] = data['y_range'][1]
        f.attrs['z_range_min'] = data['z_range'][0]
        f.attrs['z_range_max'] = data['z_range'][1]

        # 保存数据数组
        f.create_dataset('f_pbh_values', data=f_pbh_values)
        f.create_dataset('a_values', data=delta_t_grid)
        
        # 保存模型数据（保持原始结构：model_*直接在根目录）
        for model_name, model_data in models_data_new.items():
            model_group = f.create_group(model_name)  # 直接创建model_1, model_2等
            model_group.create_dataset('integral_results', data=model_data['integral_results'])
            model_group.create_dataset('N_lens', data=model_data['N_lens'])
            model_group.attrs['cosmology_type'] = model_data['cosmology_type']
            model_group.attrs['mode'] = model_data['mode']
    
    print("处理完成!")
    
    # 6. 返回处理后的数据
    result_data = {
        'R': data['R'],
        'T_obs': data['T_obs'],
        'y_range': data['y_range'],
        'z_range': data['z_range'],
        'f_pbh_values': f_pbh_values,
        'a_values': delta_t_grid,
        'models_data': models_data_new
    }
    
    return result_data

def verify_processed_data(processed_data):
    """
    验证处理后的数据
    """
    print("\n验证处理后的数据:")
    print("=" * 60)
    print(f"f_pbh值数量: {len(processed_data['f_pbh_values'])}")
    print(f"f_pbh范围: [{processed_data['f_pbh_values'].min():.2e}, {processed_data['f_pbh_values'].max():.2e}]")
    print(f"Δt值数量: {len(processed_data['a_values'])}")
    print(f"Δt范围: [{processed_data['a_values'].min():.2e}, {processed_data['a_values'].max():.2e}]")
    
    # 检查第2个模型
    first_model = list(processed_data['models_data'].keys())[1]
    model_data = processed_data['models_data'][first_model]
    results = model_data['integral_results']
    delta_t_grid = processed_data['a_values']
    
    print(f"\n模型: {first_model}")
    print(f"  cosmology_type: {model_data['cosmology_type']}")
    print(f"  mode: {model_data['mode']}")
    print(f"  integral_results 形状: {results.shape}")
    print(f"  N_lens 形状: {model_data['N_lens'].shape}")
    
    # 检查归一化
    print("\n归一化检查 (f_pbh值):")
    f_pbh_values = processed_data['f_pbh_values']
    for i in range(min(1, len(f_pbh_values))):
        integral_val = np.trapz(results[i, :], delta_t_grid)
        #print(f"p(Δt)={delta_t_grid*results[i, :]}")
        print(f"  f_pbh={f_pbh_values[i]:.3e}: ∫p(Δt)dΔt = {integral_val:.6f}")

# 使用示例
# 修改后的主程序
if __name__ == "__main__":
    input_h5 = 'data/lensing_analysis_data/bar_dt/lensing_analysis_results_1e+10_1.h5'
    
    # 指定输出目录
    output_dir = 'data/lensing_analysis_data/dt'
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(input_h5))[0]
    output_h5 = os.path.join(output_dir, f"{base_name}_dt.h5")
    
    # 处理文件
    processed_data = process_hdf5_to_p_delta_t(
        input_h5_path=input_h5,
        output_h5_path=output_h5,  # 直接使用自定义的输出路径
        n_delta_t_points=400
    )
    
    # 读取验证
    input_data = read_hdf5_data(input_h5)
    out_data = read_hdf5_data(output_h5)
    
    print(f"输入文件模型: {list(input_data['models_data'].keys())}")
    print(f"输出文件模型: {list(out_data['models_data'].keys())}")
    
    print(f"输入文件a_values形状: {input_data['a_values'].shape}")
    print(f"输出文件a_values形状: {out_data['a_values'].shape}")
    
    # 验证输出文件位置
    print(f"\n输出文件已保存到: {output_h5}")
    print(f"输出文件存在: {os.path.exists(output_h5)}")
    
    # 验证结果
    verify_processed_data(processed_data)
    
    '''
    # 可选：快速可视化第一个模型的几个分布
    import matplotlib.pyplot as plt
    
    first_model = list(processed_data['models_data'].keys())[0]
    model_data = processed_data['models_data'][first_model]
    delta_t_grid = processed_data['a_values']
    f_pbh_values = processed_data['f_pbh_values']
    
    # 选择几个代表性的f_pbh值
    indices = [0, -1]
    
    plt.figure(figsize=(10, 6))
    for idx in indices:
        f_pbh = f_pbh_values[idx]
        p_delta_t = model_data['integral_results'][idx, :]
        plt.semilogx(delta_t_grid, delta_t_grid*p_delta_t, \
                     label=f'f_pbh={f_pbh:.3e}', linewidth=1, linestyle='--')
    
    plt.xlabel('Δt')
    plt.ylabel('Δtp(Δt)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, f"{first_model}_p_delta_t_distributions.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n分布图已保存到: {plot_path}")
    plt.show()
    '''