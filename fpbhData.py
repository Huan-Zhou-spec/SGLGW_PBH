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
    
    # 定义不同的MPBH和xcl值
    MPBH_values = [1e6, 1e7, 1e8, 1e9, 1e10]  # Msun
    xcl_values = [1, 10]  # Mpc
    xi0 = 10
    
    # 测试参数范围
    a_values = np.logspace(-1, 8, 100)
    f_pbh_values = np.logspace(-7, -2, 70)
    
    # 积分区域
    y_range = (1e-3, 7.0)
    z_range = (1e-3, 7.0)
    
    # GW的事件率与观测时长
    R = 5e5    # yr^-1
    T_obs = 10 # yrs
    
    # 选择并行或顺序计算
    USE_PARALLEL = True  # 设置为False使用顺序计算
    
    # 循环所有MPBH和xcl组合
    total_combinations = len(MPBH_values) * len(xcl_values)
    current_combination = 0
    
    for MPBH in MPBH_values:
        for xcl in xcl_values:
            current_combination += 1
            print(f"\n{'='*60}")
            print(f"计算组合 {current_combination}/{total_combinations}: MPBH = {MPBH:.1e} Msun, xcl = {xcl} Mpc")
            print(f"{'='*60}")
            
            # 创建HDF5文件，文件名包含MPBH和xcl信息
            h5_filename = f'data/lensing_analysis_data/lensing_analysis_results_{MPBH:.0e}_{xcl}.h5'
            print(f"创建HDF5文件: {h5_filename}")
            
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
                            cosmo_type, mode, model, f_pbh_values, MPBH, xcl, xi0, a_values, 
                            y_range, z_range, R, T_obs
                        )
                    else:
                        N_lens_array, results_3d = compute_model_sequential(
                            cosmo_type, mode, model, f_pbh_values, MPBH, xcl, xi0, a_values, 
                            y_range, z_range, R, T_obs
                        )
                    
                    # 保存数据到HDF5
                    model_group.create_dataset('integral_results', data=results_3d)
                    model_group.create_dataset('N_lens', data=N_lens_array)
                    
                    print(f"模型 {model} 数据已保存到HDF5文件")
            
            print(f"组合 MPBH={MPBH:.1e}, xcl={xcl} 计算完成，文件已保存: {h5_filename}")
    
    # 计算总时间
    end_time = time.time()
    total_time = (end_time - start_time)/3600
    print(f"\n所有组合计算完成！总用时: {total_time:.2f} 小时")
    
    # 打印所有生成的文件
    print("\n生成的数据文件列表:")
    for MPBH in MPBH_values:
        for xcl in xcl_values:
            filename = f'data/lensing_analysis_data/lensing_analysis_results_MPBH_{MPBH:.0e}_xcl_{xcl}.h5'
            print(f"  {filename}")
        
      
        
        
        
        
        