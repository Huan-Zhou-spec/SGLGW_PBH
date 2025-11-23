#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:18:54 2025

@author: ubuntu
"""

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import numpy as np
from scipy.interpolate import interp1d


#设置宇宙学
class Cosmology:
    """astropy 版本"""
    
    def __init__(self, H0=67.66, Omega_m=0.311, Omega_l=0.689, z_max=20.0, n_points=500):
        """
        初始化
        
        参数:
        H0: 哈勃常数 (km/s/Mpc)
        Omega_m: 物质密度参数
        Omega_l: 暗能量密度参数
        """
        # 创建自定义的 FlatLambdaCDM 宇宙学
        self.astropy_cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)
        
        # 保持原有参数名
        #self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_l = Omega_l
    
        # 预计算表格
        self._precompute_tables(z_max, n_points)
    
    def Omega_m_z(self, z):
        """
        计算红移z处的物质密度参数
        """
        numerator = self.Omega_m * (1 + z)**3
        denominator = numerator + self.Omega_l
        return numerator / denominator
    
    def Delta_c(self, z, method='fixed'):
        """
        计算红移依赖的过密度参数Δc(z)
        
        参数:
        z: 红移
        method: 计算方法
            - 'fixed': 固定值200
            - 'simple': 简单公式
            - 'bryan_norman': Bryan & Norman (1998) 拟合公式
        """
        if method == 'fixed':
            return 200.0
        
        elif method == 'simple':
            # 简单公式
            Omega_m_z = self.Omega_m_z(z)
            x = Omega_m_z - 1
            return 18 * np.pi**2 + 82 * x - 39 * x**2
        
        elif method == 'bryan_norman':
            # Bryan & Norman (1998) 拟合公式
            Omega_m_z = self.Omega_m_z(z)
            x = Omega_m_z - 1
            return 18 * np.pi**2 + 60 * x - 32 * x**2
        
        else:
            raise ValueError("未知的计算方法")
    
    
    def _precompute_tables(self, z_max, n_points):
        """预计算表格"""
        
        self._z_table = np.linspace(0, z_max, n_points, dtype=float)
        
        # 使用 astropy 计算距离
        comoving_mpc = self.astropy_cosmo.comoving_distance(self._z_table)
        self._comoving_table_km = comoving_mpc.to(u.km).value
        
        # 计算哈勃参数
        Hz_quantity = self.astropy_cosmo.H(self._z_table)
        self._Hz_table_s = Hz_quantity.to(1/u.s).value
        
        # 创建插值函数
        self._comoving_interp = interp1d(
            self._z_table, self._comoving_table_km, 
            kind='linear', bounds_error=False, fill_value="extrapolate"
        )
        
        self._Hz_interp = interp1d(
            self._z_table, self._Hz_table_s,
            kind='linear', bounds_error=False, fill_value="extrapolate"
        )
        
    
    def Hz(self, z):
        """哈勃参数 (s^-1) - 完全兼容原有接口"""
        z = np.asarray(z, dtype=float)
        return self._Hz_interp(z)
    
    def _comoving_distance(self, z):
        """共动距离 (km) - 完全兼容原有接口"""
        z = np.asarray(z, dtype=float)
        return self._comoving_interp(z)
    
    def angular_diameter_distances(self, z_l, z_s):
        """
        角直径距离 (km) - 完全兼容原有接口
        返回: D_l, D_s, D_ls
        """
        z_l = np.asarray(z_l, dtype=float)
        z_s = np.asarray(z_s, dtype=float)
        
        D_c_l = self._comoving_distance(z_l)
        D_c_s = self._comoving_distance(z_s)
        
        D_l = D_c_l / (1 + z_l)
        D_s = D_c_s / (1 + z_s)
        D_ls = (D_c_s - D_c_l) / (1 + z_s)
        
        return D_l, D_s, D_ls
    
'''
cosmo = Cosmology()    
print(cosmo.Hz(0), cosmo.Delta_c(9), cosmo._comoving_distance([1,2,3]))
'''
    
    