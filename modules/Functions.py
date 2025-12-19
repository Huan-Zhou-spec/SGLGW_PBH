""""
Cosmological functions
Pablo Villanueva Domingo
Started: October 2018
Last update: April 2021
"""

import math
import numpy as np
from scipy import integrate
from .constants import *
# Employ the Colossus package for some cosmological computations:
# https://bdiemer.bitbucket.io/colossus/tutorials.html
from colossus.cosmology import cosmology
from colossus.lss import mass_function
#from colossus.halo.concentration import concentration



cosmo = cosmology.setCosmology('planck18')
rho_c_Mpc = cosmo.rho_c(0)*1e9   # in M_sun h^2 Mpc^-3

#--- POWER SPECTRUM ---#
class PBHPowerSpectrum:
    """
    PBH功率谱计算类，封装所有参数
    """
    
    def __init__(self, fpbh, Mpbh, xcl=None, xi0=None, mode='pos'):
        """
        初始化PBH功率谱计算器
        
        Parameters:
        -----------
        fpbh : float
            PBH质量分数
        Mpbh : float
            PBH质量 [Msun]
        xcl : float, optional
            聚类尺度 [Mpc]（clu模式必需）
        xi0 : float, optional
            初始两点关联函数（clu模式必需）
        mode : str, optional
            模式选择：'clu'（聚类）或 'pos'（泊松）
        """
        self.fpbh = fpbh
        self.Mpbh = Mpbh
        self.xcl = xcl
        self.xi0 = xi0
        self.mode = mode
        
        # 验证参数
        self._validate_params()
        
        # 预计算常量
        self._precompute_constants()
    
    def _validate_params(self):
        """验证参数的有效性"""
        if self.mode not in ['clu', 'pos']:
            raise ValueError("模式必须是 'clu' 或 'pos'")
            
        if self.mode == 'clu' and (self.xcl is None or self.xi0 is None):
            raise ValueError("在clu模式下，xcl和xi0参数是必需的")
    
    def _precompute_constants(self):
        """预计算常量"""
        # 宇宙学参数（假设这些已在全局定义）
        self.onepluszeq = 3401  # 1+zeq
        self.gamma = Omega_dm /Omega_m
        #self.npbh = 3.3e10 * self.fpbh / self.Mpbh / hlittle**3  #(h/Mpc)^3
        self.npbh = rho_c_Mpc*Omega_dm*self.fpbh/self.Mpbh/hlittle #(h/Mpc)^3
        
        # 模式特定的预计算
        if self.mode == 'clu':
            self.kclu =  (2 * np.pi**2 * self.npbh/ \
                          (1+self.npbh*4*np.pi/3*self.xcl**3*hlittle**3*self.xi0))**(1/3)   # from ncl/Nc, h/Mpc
            #self.kclu = (3 * np.pi / 2 / self.xi0)**(1/3) / self.xcl / hlittle #Ncl>>1
        elif self.mode == 'pos':
            self.kpbh = (2 * np.pi**2 * self.npbh)**(1/3) # from npbh, h/Mpc
    
    def isochronous_power_spectrum(self, k):
        """
        计算PBH功率谱
        
        Parameters:
        -----------
        k : float or array_like
            波数 [h/Mpc]
            
        Returns:
        --------
        Pk : float or array_like
            功率谱 [Mpc^3/h^3]
        """
        if self.mode == 'clu':
            return self._isochronous_clu(k)
        elif self.mode == 'pos':
            return self._isochronous_pos(k)
    
    def _isochronous_clu(self, k):
        """成团PBH的功率谱计算"""
        a_ = (np.sqrt(1+24*self.gamma)-1)/4
        Tpbh = (1+1.5*self.gamma/a_*self.onepluszeq)**a_
        if isinstance(k, (int, float)):
            # 单个k值
            if k <= self.kclu:
                term1 = 1 / self.npbh
                term2 =  4 * np.pi * self.xi0 / k**3 * \
                        (np.sin(k / hlittle * self.xcl) - k/hlittle * self.xcl * np.cos(k / hlittle * self.xcl))
                return   self.fpbh**2 *Tpbh**2 * (term1 + term2)
            else:
                return 0.0
        else:
            # k数组
            result = np.zeros_like(k)
            mask = k <= self.kclu
            k_masked = k[mask]
            
            term1 = 1 / self.npbh
            term2 = 4 * np.pi * self.xi0 / k_masked**3 * \
                    (np.sin(k_masked/hlittle * self.xcl) - k_masked/hlittle * self.xcl * np.cos(k_masked/hlittle * self.xcl))
            
            result[mask] = self.fpbh**2 *Tpbh**2 * (term1 + term2)
            return result
    
    def _isochronous_pos(self, k):
        """泊松PBH的功率谱计算"""
        a_ = (np.sqrt(1+24*self.gamma)-1)/4
        Tpbh = (1+1.5*self.gamma/a_*self.onepluszeq)**a_
        if isinstance(k, (int, float)):
            # 单个k值
            if k <= self.kpbh:
                term = 1 / self.npbh
                return self.fpbh**2 * Tpbh**2 * term
            else:
                return 0.0
        else:
            # k数组
            result = np.zeros_like(k)
            mask = k <= self.kpbh
            term = 1 / self.npbh
            result[mask] = self.fpbh**2 * Tpbh**2 * term
            return result
    
    def total_power_spectrum(self, k):
        """
        计算PBH宇宙中的总物质功率谱 (z=0)
        
        Parameters:
        -----------
        k : float or array_like
            波数 [h/Mpc]
            
        Returns:
        --------
        Pk : float or array_like
            总物质功率谱 [(Mpc/h)^3]
        """
        # 基础宇宙学功率谱
        Pk_cosmo = cosmo.matterPowerSpectrum(k)
        
        # PBH功率谱
        Pk_pbh = self.isochronous_power_spectrum(k)
        
        return Pk_cosmo + Pk_pbh

# 保持原有函数接口的包装函数
def IsoPS(k, fpbh, Mpbh, xcl=None, xi0=None, mode='pos'):
    """
    计算PBH功率谱
    """
    pbh_ps = PBHPowerSpectrum(fpbh, Mpbh, xcl, xi0, mode)
    return pbh_ps.isochronous_power_spectrum(k)


# Total power spectrum for PBH universes at z=0, (Mpc/h)^3
def PkPBH(k, fpbh, Mpbh, xcl=None, xi0=None, mode='pos'):
    """
    计算PBH宇宙中的总物质功率谱 (z=0)
    """
    pbh_ps = PBHPowerSpectrum(fpbh, Mpbh, xcl, xi0, mode)
    return pbh_ps.total_power_spectrum(k)



#--- HALOS ---#
#方差计算器，用于计算线性场的方差及其导数
class VarianceCalculator:
    
    def __init__(self, cutoff=10, num_k=1000):
        """
        初始化方差计算器
        
        Parameters:
        -----------
        cutoff : float, optional
            对数k积分的上下限，默认 ±10
        num_k : int, optional
            积分点数，默认 1000
        """
        self.cutoff = cutoff
        self.num_k = num_k
        self.logk_array = np.linspace(-cutoff, cutoff, num=num_k)
        
        # 预计算一些常量
        self.prefactor_sigma2 = 1.0 / (2.0 * np.pi**2)
        self.prefactor_dersigma2 = 2.0 / 3.0 * self.prefactor_sigma2
    
    @staticmethod
    def mass_to_radius(M):
        """
        从质量计算半径
        
        Parameters:
        -----------
        M : float or array_like
            质量 [M_sun/h]
            
        Returns:
        --------
        R : float or array_like
            半径 [Mpc/h]
        """
        return (4.0 * np.pi / 3.0 * Omega_m * rho_c_Mpc)**(-1.0/3.0) * M**(1.0/3.0)
    
    @staticmethod
    def window_top_hat(x):
        """
        Top-hat 窗函数
        
        Parameters:
        -----------
        x : float or array_like
            无量纲参数 k*R
            
        Returns:
        --------
        W : float or array_like
            窗函数值
        """
        x = np.asarray(x)
        # 处理 x=0 的情况
        result = np.zeros_like(x)
        mask = x != 0
        x_masked = x[mask]
        result[mask] = 3.0 * (np.sin(x_masked) - x_masked * np.cos(x_masked)) / x_masked**3
        result[~mask] = 1.0  # x=0 时的极限值
        return result
    
    @staticmethod
    def der_window_top_hat(x):
        """
        Top-hat 窗函数的对数导数 dW/dlnR
        
        Parameters:
        -----------
        x : float or array_like
            无量纲参数 k*R
            
        Returns:
        --------
        dW_dlnR : float or array_like
            窗函数的对数导数
        """
        x = np.asarray(x)
        # 处理 x=0 的情况
        result = np.zeros_like(x)
        mask = x != 0
        x_masked = x[mask]
        result[mask] = (9.0 * x_masked * np.cos(x_masked) + 
                        3.0 * (x_masked**2 - 3.0) * np.sin(x_masked)) / x_masked**3
        result[~mask] = 0.0  # x=0 时的极限值
        return result
    
    def sigma2(self, M, pbh_power_spectrum):
        """
        计算线性场的方差
        
        Parameters:
        -----------
        M : float or array_like
            质量 [M_sun/h]
        pbh_power_spectrum : PBHPowerSpectrum instance
            PBH功率谱计算器实例
            
        Returns:
        --------
        sigma2 : float or array_like
            方差值
        """
        M = np.asarray(M)
        R = self.mass_to_radius(M)
        
        if M.ndim == 0:  # 标量输入
            integrand = (self.prefactor_sigma2 * np.exp(3.0 * self.logk_array) * 
                        pbh_power_spectrum.total_power_spectrum(np.exp(self.logk_array)) * 
                        self.window_top_hat(np.exp(self.logk_array) * R)**2)
            return integrate.simpson(integrand, self.logk_array)
        else:  # 数组输入
            results = np.zeros_like(M)
            for i, (mass, radius) in enumerate(zip(M, R)):
                integrand = (self.prefactor_sigma2 * np.exp(3.0 * self.logk_array) * 
                            pbh_power_spectrum.total_power_spectrum(np.exp(self.logk_array)) * 
                            self.window_top_hat(np.exp(self.logk_array) * radius)**2)
                results[i] = integrate.simpson(integrand, self.logk_array)
            return results
    
    def der_sigma2(self, M, pbh_power_spectrum):
        """
        计算方差的导数 dσ²/dlnM
        
        Parameters:
        -----------
        M : float or array_like
            质量 [M_sun/h]
        pbh_power_spectrum : PBHPowerSpectrum instance
            PBH功率谱计算器实例
            
        Returns:
        --------
        der_sigma2 : float or array_like
            方差导数值
        """
        M = np.asarray(M)
        R = self.mass_to_radius(M)
        
        if M.ndim == 0:  # 标量输入
            integrand = (self.prefactor_dersigma2 * np.exp(3.0 * self.logk_array) * 
                        pbh_power_spectrum.total_power_spectrum(np.exp(self.logk_array)) * 
                        self.window_top_hat(np.exp(self.logk_array) * R) * 
                        self.der_window_top_hat(np.exp(self.logk_array) * R))
            return integrate.simpson(integrand, self.logk_array)
        else:  # 数组输入
            results = np.zeros_like(M)
            for i, (mass, radius) in enumerate(zip(M, R)):
                integrand = (self.prefactor_dersigma2 * np.exp(3.0 * self.logk_array) * 
                            pbh_power_spectrum.total_power_spectrum(np.exp(self.logk_array)) * 
                            self.window_top_hat(np.exp(self.logk_array) * radius) * 
                            self.der_window_top_hat(np.exp(self.logk_array) * radius))
                results[i] = integrate.simpson(integrand, self.logk_array)
            return results


def create_variance_calculator(cutoff=10, num_k=1000):
    """
    创建方差计算器
    
    Returns:
    --------
    calculator : VarianceCalculator
        方差计算器实例
    sigvec : function
        向量化的方差计算函数
    dersigvec : function
        向量化的方差导数计算函数
    """
    calculator = VarianceCalculator(cutoff, num_k)
    
    def sigvec_wrapper(M, fpbh, Mpbh, xcl, xi0, mode):
        pbh_ps = PBHPowerSpectrum(fpbh, Mpbh, xcl, xi0, mode)
        return calculator.sigma2(M, pbh_ps)
    
    def dersigvec_wrapper(M, fpbh, Mpbh, xcl, xi0, mode):
        pbh_ps = PBHPowerSpectrum(fpbh, Mpbh, xcl, xi0, mode)
        return calculator.der_sigma2(M, pbh_ps)
    
    return calculator, np.vectorize(sigvec_wrapper), np.vectorize(dersigvec_wrapper)



# Press-Schechter first crossing distribution
def f(x, model='st', **kwargs):
    """
    统一的首次穿越分布函数
    
    Parameters:
    -----------
    x : array-like
        ν = (δ_c/σ)^2
    model : str, optional
        模型选择:
        - 'press_schechter' 或 'ps': Press-Schechter 模型
        - 'sheth_tormen' 或 'st': Sheth-Tormen 模型 (默认)
        - 'jenkins' 或 'jk': Jenkins 模型
    **kwargs : dict
        模型特定参数:
        - Sheth-Tormen 模型: p, q, A
        - Jenkins 模型: A, b, c
    
    Returns:
    --------
    f_nu : array-like
        首次穿越分布函数值
    """
    x = np.asarray(x)
    
    # 模型选择映射 - 统一所有可能的名称
    model_map = {
        # Press-Schechter 模型
        'press_schechter': 'press_schechter',
        'ps': 'press_schechter',
        'press74': 'press_schechter',
        
        # Sheth-Tormen 模型  
        'sheth_tormen': 'sheth_tormen',
        'st': 'sheth_tormen',
        'sheth99': 'sheth_tormen',
        
        # Jenkins 模型
        'jenkins': 'jenkins',
        'jk': 'jenkins',
        'jenkins01': 'jenkins'
    }
    
    model_normalized = model_map.get(model.lower(), model)
    
    if model_normalized == 'press_schechter':
        return np.sqrt(2.0 * x / np.pi) * np.exp(-x / 2.0)
    
    elif model_normalized == 'sheth_tormen':
        # Sheth-Tormen 模型参数 (标准值)
        p = kwargs.get('p', 0.3)
        q = kwargs.get('q', 0.707)
        A = kwargs.get('A', 0.3222)
        
        return A * np.sqrt(2.0 * q * x / np.pi) * (1.0 + (q * x) ** (-p)) * np.exp(-q * x / 2.0)
    
    elif model_normalized == 'jenkins':
        # Jenkins 模型参数
        A = kwargs.get('A', 0.315)
        b = kwargs.get('b', 0.61) 
        c = kwargs.get('c', 3.8)
        
        sigma = 1.68647 / np.sqrt(x)  # σ = 1.68647/√ν
        ln_sigma_inv = np.log(1.0 / sigma)
        return A * np.exp(-np.abs(ln_sigma_inv + b) ** c)
    
    else:
        raise ValueError(f"未知模型: {model}。可选模型: 'press_schechter'/'ps'/'press74', 'sheth_tormen'/'st'/'sheth99', 'jenkins'/'jk'/'jenkins01'")


# Halo mass function for CDM and PBH scenarios, dndlnM, units of (h/Mpc)^3
def halo_mass_function(M, z, cosmology_type='CDM',  model='st', **kwargs):
    """
    统一的晕质量函数计算
    
    Parameters:
    -----------
    M : array-like
        质量数组 [M_sun/h]
    z : array-like
        红移数组
    cosmology_type : str
        宇宙学类型: 'CDM' (默认) 或 'PBH'
    model : str
        质量函数模型: 
        - 对于 CDM: 'press74', 'sheth99', 'jenkins01' (colossus 模型名称)
        - 对于 PBH+CDM: 'press_schechter', 'sheth_tormen', 'jenkins' (自定义模型名称)
    **kwargs : dict
        额外参数:
        - 对于 PBH: fpbh, Mpbh, xcl, xi0, mode
        - 模型特定参数
    
    Returns:
    --------
    dndlnM : array-like
        质量函数值 [(h/Mpc)^3]
    """
    M = np.asarray(M)
    
    # 宇宙学类型映射
    cosmology_map = {
        'CDM': 'CDM',
        'CDM+PBH': 'CDM+PBH', 
        'PBH': 'CDM+PBH'  # 将 PBH 映射为 CDM+PBH 以保持一致性
    }
    cosmology_normalized = cosmology_map.get(cosmology_type.upper(), cosmology_type)
    
    if cosmology_normalized == 'CDM':
        # CDM 宇宙学的质量函数 - 使用 colossus 库
        try:
            # colossus 模型名称映射
            colossus_model_map = {
                'press_schechter': 'press74',
                'ps': 'press74',
                'press74': 'press74',
                'sheth_tormen': 'sheth99', 
                'st': 'sheth99',
                'sheth99': 'sheth99',
                'jenkins': 'jenkins01',
                'jk': 'jenkins01',
                'jenkins01': 'jenkins01'
            }
            
            colossus_model = colossus_model_map.get(model.lower(), model)
            
            # 检查模型是否有效
            valid_models = ['press74', 'sheth99', 'jenkins01']
            if colossus_model not in valid_models:
                print(f"警告: CDM 宇宙学中模型 '{model}' 可能无效，使用默认模型 'sheth99'")
                colossus_model = 'sheth99'
            
            return mass_function.massFunction(M, z, q_out="dndlnM", model=colossus_model)
            
        except Exception as e:
            print(f"使用 colossus 计算 CDM 质量函数时出错: {e}")
            print("回退到自定义计算...")
            # 回退到自定义计算
            calculator, sigvec, dersigvec = create_variance_calculator()
            s2 = sigvec(M)
            deltac = 1.68647 / cosmo.growthFactor(z)
            nu2 = (deltac**2 / s2)
            dlogsigdlogm = np.abs(dersigvec(M) / (2. * s2))
            
            # 使用自定义 f 函数
            return Omega_m * rho_c_Mpc * f(nu2, model=model, **kwargs) / M * dlogsigdlogm
    
    elif cosmology_normalized == 'CDM+PBH':
        # PBH 宇宙学的质量函数 - 使用自定义计算
        calculator, sigvec, dersigvec = create_variance_calculator()
        
        # 提取 PBH 特定参数
        fpbh = kwargs.get('fpbh', 1e-3)
        Mpbh = kwargs.get('Mpbh', 1e9)
        xcl = kwargs.get('xcl', 1.0)
        xi0 = kwargs.get('xi0', 1.0)
        mode = kwargs.get('mode', 'default')
        
        # 计算基本量
        s2 = sigvec(M, fpbh, Mpbh, xcl, xi0, mode)
        deltac = 1.68647 / cosmo.growthFactor(z)
        nu = (deltac**2 / s2)
        dlogsigdlogm = np.abs(dersigvec(M, fpbh, Mpbh, xcl, xi0, mode) / (2. * s2))
        
        # 使用统一的 f 函数计算
        return Omega_m * rho_c_Mpc * f(nu, model=model, **kwargs) / M * dlogsigdlogm
    
    else:
        raise ValueError(f"未知宇宙学类型: {cosmology_type}。可选: 'CDM', 'CDM+PBH'")


def test_halo_mass_function_v(sigma,zl):
    n0=100
    alpha = 6.75
    beta = 2.37
    sigma_1 = 119
    n = n0*(1+zl)**3*(sigma/sigma_1)**alpha*np.exp(-(sigma/sigma_1)**beta)/sigma*beta\
        /math.gamma(alpha/beta)
    return n


# 辅助函数：获取可用的模型列表
def get_available_models(cosmology_type='CDM'):
    """
    获取指定宇宙学类型下可用的模型列表
    
    Parameters:
    -----------
    cosmology_type : str
        宇宙学类型: 'CDM' 或 'CDM+PBH'
    
    Returns:
    --------
    models : list
        可用的模型名称列表
    """
    cosmology_map = {
        'CDM': 'CDM',
        'CDM+PBH': 'CDM+PBH', 
        'PBH': 'CDM+PBH'
    }
    cosmology_normalized = cosmology_map.get(cosmology_type.upper(), cosmology_type)
    
    if cosmology_normalized == 'CDM':
        return ['press74', 'sheth99', 'jenkins01']
    elif cosmology_normalized == 'CDM+PBH':
        return ['press_schechter', 'sheth_tormen', 'jenkins']
    else:
        return []


#--- MISC ---#
# Write a number in Latex scientific notation
def scinot(x):
    exp = int(math.floor(math.log10(abs(x))))
    prefactor = x / 10**exp
    if exp==0:
        return r"{:.0f}".format(prefactor)
    if exp==1:
        return r"{:.0f}".format(prefactor*10.)
    elif prefactor == 1.:
        return r"$10^{"+str(exp)+"}$"
    else:
        return r"${:.0f}".format(prefactor)+" \\times 10^{"+str(exp)+"}$"


'''
#测试程序
def test_halo_mass_function_vectorization():
    """测试halo_mass_function的向量化输入能力"""
    
    # 测试用例1：标量输入
    print("测试1: 标量输入")
    try:
        result_scalar = halo_mass_function(1e10, 0.5, 
                                         cosmology_type='CDM',
                                         model='st',
                                         fpbh=1e-3,
                                         Mpbh=1e9,
                                         xcl=1.0,
                                         xi0=1.0,
                                         mode='default')
        print(f"标量输入结果: {result_scalar}")
    except Exception as e:
        print(f"标量输入失败: {e}")
        return False
    
    # 测试用例2：数组输入
    print("\n测试2: 数组输入")
    try:
        m_array = np.array([1e10, 1e11, 1e12])
        z_array = np.array([0.5, 0.5, 0.5])  # 相同红移
        
        result_array = halo_mass_function(m_array, z_array,
                                        cosmology_type='CDM',
                                        model='st',
                                        fpbh=1e-3,
                                        Mpbh=1e9,
                                        xcl=1.0,
                                        xi0=1.0,
                                        mode='default')
        print(f"数组输入结果形状: {result_array.shape}")
        print(f"数组输入结果: {result_array}")
    except Exception as e:
        print(f"数组输入失败: {e}")
        return False
    
    # 测试用例3：不同红移的数组
    print("\n测试3: 不同红移的数组")
    try:
        m_array2 = np.array([1e10, 1e11, 1e12])
        z_array2 = np.array([0.1, 0.5, 1.0])  # 不同红移
        
        result_array2 = halo_mass_function(m_array2, z_array2,
                                         cosmology_type='CDM',
                                         model='st',
                                         fpbh=1e-3,
                                         Mpbh=1e9,
                                         xcl=1.0,
                                         xi0=1.0,
                                         mode='default')
        print(f"不同红移数组结果形状: {result_array2.shape}")
        print(f"不同红移数组结果: {result_array2}")
    except Exception as e:
        print(f"不同红移数组输入失败: {e}")
        return False
    
    # 测试用例4：大规模数组
    print("\n测试4: 大规模数组")
    try:
        n_test = 1000
        m_large = np.logspace(8, 15, n_test)
        z_large = np.random.uniform(0.1, 5.0, n_test)
        
        result_large = halo_mass_function(m_large, z_large,
                                        cosmology_type='CDM',
                                        model='st',
                                        fpbh=1e-3,
                                        Mpbh=1e9,
                                        xcl=1.0,
                                        xi0=1.0,
                                        mode='default')
        print(f"大规模数组结果形状: {result_large.shape}")
        print(f"大规模数组结果范围: {np.min(result_large):.2e} 到 {np.max(result_large):.2e}")
    except Exception as e:
        print(f"大规模数组输入失败: {e}")
        return False
    
    return True

# 运行测试
if __name__ == "__main__":
    success = test_halo_mass_function_vectorization()
    print(f"\n测试结果: {'通过' if success else '失败'}")
'''























