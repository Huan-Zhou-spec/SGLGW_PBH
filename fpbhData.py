#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 21:19:34 2025

@author: ubuntu
"""

from modules import DeltaFunctionIntegral
from modules import LensingAnalysis
import numpy as np
import time
import h5py
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor



def compute_single_fpbh_point(args):
    """
    计算单个f_pbh参数点的所有a_values，用于并行化
    """
    # 解包参数，第一个是索引
    idx, cosmo_type, mode, f_pbh, M_PBH, x_cl, xi_0, a_values, y_range, z_range, R, T_obs = args
    
    # 初始化对应宇宙学模型的分析（每个f_pbh只初始化一次）
    analysis = LensingAnalysis(cosmology_type=cosmo_type, mode=mode,\
                               fpbh=f_pbh, Mpbh = M_PBH, xcl=x_cl, xi0=xi_0)
    K_func = analysis.kernel.K
    K_deriv_y = analysis.kernel.K_zl
    K_deriv_z = analysis.kernel.K_zs
    g_func = analysis.g
    
    # 计算透镜个数（每个f_pbh只计算一次）
    N_lens = analysis.compute_N_lens(R, T_obs)
    
    # 计算该f_pbh对应的所有a_values的积分结果
    results_for_fpbh = []
    for a in a_values:
        integrator = DeltaFunctionIntegral(a, K_func, K_deriv_y, K_deriv_z)
        result = integrator.compute_integral_vectorized(g_func, y_range, z_range, n_points=100)
        results_for_fpbh.append(result)
    
    # 返回索引，以便在调用函数中正确放置结果
    return idx, N_lens, np.array(results_for_fpbh)


def compute_model_parallel(cosmo_type, mode, model, f_pbh_values, M_PBH, x_cl, xi_0, a_values, y_range, z_range, R, T_obs):
    """
    并行计算单个模型的所有参数点
    """
    
    # 准备参数列表：每个任务处理一个f_pbh的所有a_values
    params_list = []
    for idx, f_pbh in enumerate(f_pbh_values):
        params_list.append((
            idx, cosmo_type, mode, f_pbh, M_PBH, x_cl, xi_0, a_values, 
            y_range, z_range, R, T_obs
        ))
    
    # 使用进程池并行计算（按f_pbh并行）
    num_processes = min(20, len(f_pbh_values))
    print(f"使用 {num_processes} 个进程进行并行计算，每个进程处理一个f_pbh的 {len(a_values)} 个a值")
    
    N_lens_array = np.zeros(len(f_pbh_values))
    results_3d = np.zeros((len(f_pbh_values), len(a_values)))
    
    with Pool(processes=num_processes) as pool:
        # 使用imap_unordered获取结果
        for result in pool.imap_unordered(compute_single_fpbh_point, params_list):
            # 解包结果，包括索引
            idx, N_lens, results = result
            
            if idx % 10 == 0:  # 每10个f_pbh点打印一次进度
                print(f"  模型 {model}: 已完成 {idx+1}/{len(f_pbh_values)} 个f_pbh点")
            
            N_lens_array[idx] = N_lens
            results_3d[idx, :] = results
    
    return N_lens_array, results_3d


def compute_model_sequential(cosmo_type, mode, model, f_pbh_values, M_PBH, x_cl, xi_0, a_values, y_range, z_range, R, T_obs):
    """
    顺序计算单个模型的所有参数点
    """

    N_lens_array = []
    results_3d = []
    
    for j, f_pbh in enumerate(f_pbh_values):
        if j % 10 == 0:
            print(f"  计算 f_pbh = {f_pbh:.2e} ({j+1}/{len(f_pbh_values)})...")
        
        # 初始化对应宇宙学模型的分析
        analysis = LensingAnalysis(cosmology_type=cosmo_type, mode=mode,\
                                   fpbh=f_pbh, Mpbh = M_PBH, xcl=x_cl, xi0=xi_0)
        K_func = analysis.kernel.K
        K_deriv_y = analysis.kernel.K_zl
        K_deriv_z = analysis.kernel.K_zs
        g_func = analysis.g
        
        # 计算透镜个数
        N_lens = analysis.compute_N_lens(R, T_obs)
        N_lens_array.append(N_lens)
        
        # 计算该f_pbh对应的所有a_values
        model_results = []
        for i, a in enumerate(a_values):
            integrator = DeltaFunctionIntegral(a, K_func, K_deriv_y, K_deriv_z)
            result = integrator.compute_integral_vectorized(g_func, y_range, z_range, n_points=100)
            model_results.append(result)
        
        results_3d.append(model_results)
    
    return np.array(N_lens_array), np.array(results_3d)


# 可选：如果a_values数量很大，也可以考虑在f_pbh内部对a_values进行并行化
def compute_single_fpbh_point_with_inner_parallel(args):
    """
    在f_pbh内部也对a_values进行并行计算（适用于a_values数量很大的情况）
    """
    cosmo_type, mode, f_pbh, M_PBH, x_cl, xi_0, a_values, y_range, z_range, R, T_obs = args
    
    # 初始化分析实例
    analysis = LensingAnalysis(cosmology_type=cosmo_type, mode=mode,\
                               fpbh=f_pbh, Mpbh = M_PBH, xcl=x_cl, xi0=xi_0)
    K_func = analysis.kernel.K
    K_deriv_y = analysis.kernel.K_zl
    K_deriv_z = analysis.kernel.K_zs
    g_func = analysis.g
    
    N_lens = analysis.compute_N_lens(R, T_obs)
    
    # 内部并行计算a_values
    def compute_single_a(a):
        integrator = DeltaFunctionIntegral(a, K_func, K_deriv_y, K_deriv_z)
        return integrator.compute_integral_vectorized(g_func, y_range, z_range, n_points=100)
    
    # 使用线程池并行计算a_values（注意：这里是线程池，因为可能是I/O密集型）
    with ThreadPoolExecutor(max_workers=min(len(a_values), cpu_count())) as executor:
        results = list(executor.map(compute_single_a, a_values))
    
    return N_lens, np.array(results)



if __name__ == "__main__":
    start_time = time.time()
    
    # 定义宇宙学模型列表
    cosmology_types = ['CDM', 'CDM+PBH', 'CDM+PBH']
    modes = ['pos', 'pos', 'clu']
    models = [1, 2, 3]
    MPBH = 1e9 #Msun
    xcl = 1 #Mpc
    xi0 = 10
    
    # 测试参数范围
    a_values = np.logspace(-1, 8, 180)
    f_pbh_values = np.logspace(-7, -2, 100)
    
    # 积分区域
    y_range = (1e-3, 7.0)
    z_range = (1e-3, 7.0)
    
    # GW的事件率与观测时长
    R = 5e5    # yr^-1
    T_obs = 10 # yr
    
    
    # 创建HDF5文件
    h5_filename = 'data/lensing_analysis_data/lensing_analysis_results.h5'
    print(f"创建HDF5文件: {h5_filename}")
    
    # 选择并行或顺序计算
    USE_PARALLEL = True  # 设置为False使用顺序计算
    
    with h5py.File(h5_filename, 'w') as h5f:
        # 保存公共参数
        h5f.attrs['R'] = R
        h5f.attrs['T_obs'] = T_obs
        h5f.attrs['MPBH'] = MPBH
        h5f.attrs['xcl'] = xcl
        h5f.attrs['xi0'] = xi0
        h5f.attrs['y_range_min'] = y_range[0]
        h5f.attrs['y_range_max'] = y_range[1]
        h5f.attrs['z_range_min'] = z_range[0]
        h5f.attrs['z_range_max'] = z_range[1]
        h5f.attrs['use_parallel'] = USE_PARALLEL
        
        # 保存参数数组
        h5f.create_dataset('a_values', data=a_values)
        h5f.create_dataset('f_pbh_values', data=f_pbh_values)
        
        print("开始计算多个模型和f_pbh值的积分...")
        
        for cosmo_type, mode, model in zip(cosmology_types, modes, models):
            print(f"\n=== 计算 {cosmo_type} 模型 (mode: {mode}, model: {model}) ===")
            
            # 为每个模型创建组
            model_group = h5f.create_group(f'model_{model}')
            model_group.attrs['cosmology_type'] = cosmo_type
            model_group.attrs['mode'] = mode
            
            # 选择计算方法
            if USE_PARALLEL:
                N_lens_array, results_3d = compute_model_parallel(
                    cosmo_type, mode, model, f_pbh_values, MPBH, xcl, xi0,a_values, 
                    y_range, z_range, R, T_obs
                )
            else:
                N_lens_array, results_3d = compute_model_sequential(
                    cosmo_type, mode, model, f_pbh_values, MPBH, xcl, xi0,a_values, 
                    y_range, z_range, R, T_obs
                )
            
            # 保存数据到HDF5
            model_group.create_dataset('integral_results', data=results_3d)
            model_group.create_dataset('N_lens', data=N_lens_array)
            
            print(f"模型 {model} 数据已保存到HDF5文件")
    
    # 计算总时间
    end_time = time.time()
    total_time = (end_time - start_time)/3600
    print(f"\n计算完成！总用时: {total_time:.2f} 小时")
    print(f"所有数据已保存到: {h5_filename}")
    
    # 验证HDF5文件内容
    print("\nHDF5文件结构:")
    with h5py.File(h5_filename, 'r') as h5f:
        def print_h5_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
        
        h5f.visititems(print_h5_structure)
        
        
      
        
        
        
        
        