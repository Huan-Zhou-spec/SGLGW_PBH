#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 10:12:27 2025

@author: ubuntu
"""

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import simpson


#读取数据
def read_hdf5_data(h5_filename):
    """
    Read data from HDF5 file
    """
    with h5py.File(h5_filename, 'r') as h5f:
        # Read common parameters
        R = h5f.attrs['R']
        T_obs = h5f.attrs['T_obs']
        y_range = (h5f.attrs['y_range_min'], h5f.attrs['y_range_max'])
        z_range = (h5f.attrs['z_range_min'], h5f.attrs['z_range_max'])
        
        # Read parameter arrays
        a_values = h5f['a_values'][:]
        f_pbh_values = h5f['f_pbh_values'][:]
        

        # Read model data
        models_data = {}
        for model_name in h5f.keys():
            if model_name.startswith('model_'):
                model_group = h5f[model_name]
                cosmo_type = model_group.attrs['cosmology_type']
                mode = model_group.attrs['mode']
                
                integral_results = model_group['integral_results'][:]
                N_lens = model_group['N_lens'][:]
                
                models_data[model_name] = {
                    'cosmology_type': cosmo_type,
                    'mode': mode,
                    'integral_results': integral_results,
                    'N_lens': N_lens
                }
                
                '''
                print(f"\n{model_name}:")
                print(f"  Cosmology type: {cosmo_type}, Mode: {mode}")
                print(f"  Integral results shape: {integral_results.shape}")
                print(f"  N_lens shape: {N_lens.shape}")
                '''
        return {
            'R': R,
            'T_obs': T_obs,
            'y_range': y_range,
            'z_range': z_range,
            'a_values': a_values,
            'f_pbh_values': f_pbh_values,
            'models_data': models_data
        }


#创建插值函数
def create_interpolators(data):
    """
    为所有模型创建插值函数
    """
    # 获取基础网格点
    f_pbh_values = data['f_pbh_values']  # 形状 (n_fpbh,)
    a_values = data['a_values']          # 形状 (n_a,)
    
    interpolators = {}
    
    for model_name, model_data in data['models_data'].items():
        
        #print(f"\n为 {model_name} 创建插值函数...")
        
        # 1. 为 integral_results 创建二维插值
        integral_results = model_data['integral_results']  # 形状 (60, 120)
        
        # 创建二维插值函数
        # 注意：RegularGridInterpolator 需要 (x, y) 格式，这里 f_pbh 是第一个维度，a 是第二个维度
        integral_interpolator = RegularGridInterpolator(
            (f_pbh_values, a_values), 
            integral_results,
            method='cubic',  # 线性插值，也可以选择 'cubic' 或 'nearest'
            bounds_error=False,
            fill_value=0.0
        )
        
        # 2. 为 N_lens 创建一维插值
        N_lens = model_data['N_lens']  # 形状 (n_fpbh,)
        
        # 创建一维插值函数
        N_lens_interpolator = interp1d(
            f_pbh_values,
            N_lens,
            kind='cubic',  # 线性插值，也可以选择 'linear' 或 'quadratic'
            bounds_error=False,
            fill_value='extrapolate'  # 允许外推
        )
        
        interpolators[model_name] = {
            'integral': integral_interpolator,
            'N_lens': N_lens_interpolator,
            'cosmology_type': model_data['cosmology_type'],
            'mode': model_data['mode']
        }
        
        
        #print(f"  integral 插值器创建完成，输入维度: f_pbh × a = {len(f_pbh_values)} × {len(a_values)}")
        #print(f"  N_lens 插值器创建完成，输入维度: {len(f_pbh_values)}")
        
    return interpolators


def compute_interpolated_data(h5_filename, multipliers_T, new_f_pbh_values=None, new_a_values=None):
    """
    为后续计算归一化的截断插值 p(\Delta|\Omega,T_obs)和归一化的截断透镜化事件数N_lens(\Omega,T_obs)
    并将大于观测时长的时间点对应的插值设为0，
    然后对 integral_dense_truncated * np.log(10) * new_a_values 沿 log10(new_a_values) 维度积分，
    并与 N_lens_dense 相乘，最后所有输出数据都除以 T_obs_hours
    
    参数:
    - h5_filename: HDF5数据文件路径
    - new_f_pbh_values: 新的 f_pbh 值数组 (默认使用原始数据的密集版本)
    - new_a_values: 新的时间值数组 (默认使用原始数据的密集版本)
    
    返回:
    - interpolated_data: 包含所有模型插值数据和积分结果的字典
    """
    # 读取原始数据
    data = read_hdf5_data(h5_filename)
    orig_a_values = data['a_values']
    orig_f_pbh_values = data['f_pbh_values']
    models_data = data['models_data']
    R = data['R']
    T_obs = data['T_obs']  # 观测时长（年）
    
    # 计算可控观测时长对应的小时数
    T_obs_hours = T_obs * 365 * 24 * multipliers_T
    #print(f"观测时长 T_obs = {T_obs} 年，对应 {T_obs_hours} 小时")
    
    
    # 创建插值器
    interpolators = create_interpolators(data)
    
    # 设置默认的新数组（如果未提供）
    if new_f_pbh_values is None:
        new_f_pbh_values = np.logspace(np.log10(orig_f_pbh_values[0]), 
                                     np.log10(orig_f_pbh_values[-1]), 500)
        
    if new_a_values is None:
        new_a_values = np.logspace(np.log10(orig_a_values[0]), 
                                 np.log10(orig_a_values[-1]), 500)
    
    # 创建掩码：标记哪些时间点小于等于观测时长
    time_mask = new_a_values <= T_obs_hours
    #print(f"在 {len(new_a_values)} 个时间点中，有 {np.sum(time_mask)} 个点小于等于观测时长")
    
    # 存储所有模型的插值结果
    interpolated_data = {
        'new_f_pbh_values': new_f_pbh_values,
        'new_a_values': new_a_values,
        'R': R,
        'T_obs_hours': T_obs_hours,
        'time_mask': time_mask,  # 保存时间掩码，便于后续使用
        'models': {}
    }
    
    # 为每个模型计算插值数据
    for model_name, model_data in models_data.items():
        interp_dict = interpolators[model_name]
        cosmo_type = model_data['cosmology_type']
        mode = model_data['mode']
        
        # 计算 integral_dense 的二维插值
        # 创建网格点
        f_pbh_grid, a_grid = np.meshgrid(new_f_pbh_values, new_a_values, indexing='ij')
        points = np.column_stack([f_pbh_grid.ravel(), a_grid.ravel()])
        
        # 计算插值
        integral_dense = interp_dict['integral'](points)
        integral_dense = integral_dense.reshape(len(new_f_pbh_values), len(new_a_values))
        
        # 应用观测时长限制：将大于观测时长的时间点对应的值设为0
        # 注意：integral_dense 的形状是 (f_pbh, a)，所以我们在第二个维度上应用掩码
        integral_dense_truncated = integral_dense.copy()
        integral_dense_truncated[:, ~time_mask] = 0
        
        # 计算 N_lens_dense 的一维插值
        N_lens_dense = interp_dict['N_lens'](new_f_pbh_values) * multipliers_T
        
        # 创建 log10(new_a_values) 作为积分变量
        log_a_values = np.log10(new_a_values)
        
        # 计算被积函数：integral_dense_truncated * np.log(10) * new_a_values
        # 注意：new_a_values 需要扩展为与 integral_dense_truncated 相同的形状
        a_values_expanded = np.tile(new_a_values, (len(new_f_pbh_values), 1))
        integrand = integral_dense_truncated * np.log(10) * a_values_expanded *\
            (T_obs_hours-a_values_expanded)
        
        #截断后所对应的p(\Delta|\Omega,T_obs)
        integrand_dense = integral_dense_truncated * (T_obs_hours-a_values_expanded)
        
        # 对 integrand 沿着 log10(new_a_values) 维度进行积分
        # 使用 Simpson 积分法，axis=1 表示沿着第二个维度（时间维度）积分：S(\Omega,T_obs)
        integral_over_time = simpson(integrand, log_a_values, axis=1)
        
        # 将积分结果与 N_lens_dense 相乘
        N_lens_multiplied = integral_over_time * N_lens_dense
        
        # 将所有输出数据除以 T_obs_hours并归一化
        integrand_dense_normalized = integrand_dense/integral_over_time[:,np.newaxis]
        N_lens_multiplied_normalized = N_lens_multiplied / T_obs_hours
        
        # 存储结果（使用归一化后的数据）
        interpolated_data['models'][model_name] = {
            'cosmology_type': cosmo_type,
            'mode': mode,
            'integral_dense': integral_dense,  # 原始插值 p(\Delta|\Omega)
            'integrand_dense': integrand_dense_normalized,  # 归一化的截断插值 p(\Delta|\Omega,T_obs)
            'N_lens_dense': N_lens_dense,       # 原始透镜事件数 N_lens(\Omega)
            'N_lens_multiplied': N_lens_multiplied_normalized  # 归一化的截断透镜化事件数N_lens(\Omega,T_obs)
        }
        
        #print(f"{model_name} ({cosmo_type} {mode}):")
        #print(f"  integral_dense 形状: {integral_dense_normalized.shape}")
        #print(f"  N_lens_dense 形状: {N_lens_dense_normalized.shape}")
    
    return interpolated_data


   

    
    
    
    
    
    
    