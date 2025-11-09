# %%
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from .Cosmo import Cosmology
from .constants import *
from .Functions import halo_mass_function
from scipy.interpolate import interp1d, RectBivariateSpline

# 设置宇宙学
cosmo = Cosmology()
#速度弥散与暗物质晕之间的转化公式
def sis_velocity_dispersion(z, M=None, sigma=None):
    """
    SIS模型速度弥散计算器
    
    参数:
    z: 红移
    M: 暗物质晕质量 (Msun/h) - 提供此项计算速度弥散
    sigma: 速度弥散 (km/s) - 提供此项计算质量
    
    返回:
    速度弥散 (km/s) 或 质量 (Msun/h)
    
    具体计算过程
    # 计算临界密度 (g/cm³)
    rho_crit = 3 * cosmo.Hz(z)**2 / (8 * np.pi * G)
    
    # 计算维里半径 R200 (cm)
    delta = 200  # 超密度对比
    R200_cm = (3 * M_g / (4 * np.pi * delta * rho_crit))**(1/3)
    
    # 使用SIS模型计算速度弥散: σ = √[G * M / (R)]
    sigma_km_s = 1e-5*np.sqrt(G * M_g / ( R200_cm))  # km/s
    
    """
    f = 1
    if M is not None:
        # 计算速度弥散
        M_g = M * Msun * hlittle
        return f* 1e-5 * (np.sqrt(cosmo.Delta_c(z)/2) * G * M_g * cosmo.Hz(z))**(1/3)
        
    elif sigma is not None:
        # 计算质量
        sigma_cm_s = sigma * 1e5
        return sigma_cm_s**3 / np.sqrt(cosmo.Delta_c(z)/2)\
             / G / cosmo.Hz(z) / Msun / hlittle/f**3
        
    else:
        raise ValueError("必须提供M或sigma中的一个")

#print(sis_velocity_dispersion(0, 1e16, None))


#SIS透镜分析
class LensingAnalysis:
    """
    引力透镜分析类，整合所有相关函数
    """
    
    def __init__(self, pz_data_file="data/pz.txt", precompute_tau=True, **kwargs):
        """
        初始化
        
        参数:
        cosmology: 宇宙学对象
        pz_data_file: 红移分布数据文件路径
        precompute_tau: 是否预计算光学深度积分
        **kwargs: 所有计算参数
        """
        self.cosmo = cosmo
        self.c_km = c / 1e5  # 光速 (km/s)
        self.Mpc_h_km = MpcToCm / 1e5 / hlittle
        self.mmin = 1e10*hlittle #Msun/h
        self.mmax = 1e15*hlittle 
        self.zmin = 1e-3
        self.zmax = 7
        
        # 设置默认参数并更新用户提供的参数
        self.parameters = {
            'fpbh': 1e-3,
            'Mpbh': 1e9,
            'xcl': 1.0,
            'xi0': 100,
            'cosmology_type': 'CDM',
            'model': 'st',
            'mode': 'pos'
        }
        self.parameters.update(kwargs)
        
        # 加载红移分布数据
        self._load_pz_data(pz_data_file)
        
        # 初始化透镜核类
        self.kernel = self.LensingKernel(cosmo)
        
        # 预计算光学深度积分
        self.tau_interpolator = None
        
        if precompute_tau:
            self._precompute_tau_integral()
    
    def _load_pz_data(self, pz_data_file):
        """加载红移分布数据"""
        try:
            ksa, zsa, psa = np.loadtxt(pz_data_file, unpack=True)
            self.zsa = zsa
            self.psa = psa
        except Exception as e:
            print(f"警告: 无法加载红移分布数据 {pz_data_file}: {e}")
            # 设置默认值
            self.zsa = np.array([0, 1, 2, 3])
            self.psa = np.array([1, 0.5, 0.1, 0.01])
    
    
    def _precompute_tau_integral(self, zs_points=50, sigma_points=200, zl_points=100):
        """
        预计算光学深度积分并创建插值器
        
        参数:
        zs_points: 源红移采样点数
        sigma_points: sigma采样点数  
        zl_points: zl采样点数
        """
        print("预计算光学深度积分...")
        
        # 定义源红移范围
        zs_min = self.zmin
        zs_max = self.zmax
        zs_values = np.linspace(zs_min, zs_max, zs_points)
        
        # 预计算每个zs的积分值
        tau_values = np.zeros(zs_points)
        
        for i, zs in enumerate(zs_values):
            tau_values[i] = self._compute_tau_integral_hybrid(zs, sigma_points, zl_points)
            if i % 10 == 0:
                print(f"进度: {i+1}/{zs_points}, zs={zs:.2f}, tau={tau_values[i]:.2e}")
        
        # 创建插值函数
        self.tau_interpolator = interp1d(zs_values, tau_values, 
                                       kind='cubic', 
                                       bounds_error=False, 
                                       fill_value=0.0)
        
        print(f"光学深度积分预计算完成，参数: {self.parameters}")
    
    def _compute_tau_integral(self, zs, sigma_points, zl_points):
        """计算单个zs的光学深度积分"""
        Mmin, Mmax = self.mmin, self.mmax  #Msun/h
        zl_min = 1e-4
        zl_max = max(zl_min, zs)
        sigma_min, sigma_max = sis_velocity_dispersion(zl_min,Mmin), \
            sis_velocity_dispersion(zs, Mmax) #km/s
        
        # 创建网格
        #sigma_grid = np.linspace(sigma_min, sigma_max, sigma_points)
        sigma_grid = np.logspace(np.log10(sigma_min), np.log10(sigma_max), sigma_points)
        zl_grid = np.linspace(zl_min, zl_max, zl_points)
        log_sigma_grid = np.log10(sigma_grid)
        
        # 创建二维坐标数组（展平）
        sigma_flat = np.repeat(sigma_grid[:, np.newaxis], zl_points, axis=1).flatten()
        zl_flat = np.repeat(zl_grid[np.newaxis, :], sigma_points, axis=0).flatten()
        
        # 向量化计算（使用类参数）
        integrand_flat = self.ptau(sigma_flat, zl_flat, zs)*sigma_flat*np.log(10)
        
        # 重塑为二维数组
        integrand_values = integrand_flat.reshape(sigma_points, zl_points)
        
        # 积分
        tau = np.trapz(np.trapz(integrand_values, zl_grid, axis=1), log_sigma_grid)
        
        return tau

    def _compute_tau_integral_hybrid(self, zs, sigma_points=100, zl_points=100):
        """混合方法：固定网格 + 重要性采样"""
        Mmin, Mmax = self.mmin, self.mmax
        zl_min = 1e-4
        zl_max = zs
    
        # 创建优化的网格
        sigma_grid = self._create_optimized_sigma_grid(zs, sigma_points)
        zl_grid = np.linspace(zl_min, zl_max, zl_points)
    
        # 使用meshgrid避免重复计算
        sigma_2d, zl_2d = np.meshgrid(sigma_grid, zl_grid, indexing='ij')
    
        # 批量计算
        integrand = self.ptau(sigma_2d, zl_2d, zs)*sigma_2d
    
        # 积分
        tau_zl = np.trapz(integrand, zl_grid, axis=1)
        tau = np.trapz(tau_zl, np.log(sigma_grid))  # 在对数空间积分
    
        return tau
    

    def _create_optimized_sigma_grid(self, zs, n_points):
        """创建优化的σ网格，在变化剧烈的区域加密"""
        Mmin, Mmax = self.mmin, self.mmax
        sigma_min = sis_velocity_dispersion(1e-4, Mmin)
        sigma_max = sis_velocity_dispersion(zs, Mmax)
    
        # 基础对数网格
        log_sigma = np.linspace(np.log(sigma_min), np.log(sigma_max), n_points)
    
        # 可以在特定区域进一步加密（如果需要）
        # 例如在函数变化剧烈的地方增加点数
    
        return np.exp(log_sigma)


    class LensingKernel:
        """
        透镜核函数类
        """
        
        def __init__(self, cosmology):
            self.cosmo = cosmology
            self.c_km = c / 1e5
            self.const_base = 32 * np.pi**2/3600
        
        def get_all_kernels(self, zl_array, zs_array):
            """
            一次性计算所有核函数
            
            返回:
            K, K_zl, K_zs #hrs/(km/s)^4
            """
            Dl = self.cosmo._comoving_distance(zl_array)
            Ds = self.cosmo._comoving_distance(zs_array)
            Dls = Ds - Dl
            Hz_l = self.cosmo.Hz(zl_array)
            Hz_s = self.cosmo.Hz(zs_array)
            
            c4_inv = (1/self.c_km)**4
            c5_inv = c4_inv / self.c_km
            const = self.const_base
            
            K = const * c5_inv * Dls * Dl / Ds
            K_zl = const * c4_inv  * (Ds - 2*Dl) / Ds / Hz_l
            K_zs = const * c4_inv * (Dl**2 / Ds**2 / Hz_s)
            
            return K, K_zl, K_zs
        
        def K(self, zl_array, zs_array):
            """主核函数"""
            K_val, _, _ = self.get_all_kernels(zl_array, zs_array)
            return K_val
        
        def K_zl(self, zl_array, zs_array):
            """对zl的导数"""
            _, K_zl_val, _ = self.get_all_kernels(zl_array, zs_array)
            return K_zl_val
        
        def K_zs(self, zl_array, zs_array):
            """对zs的导数"""
            _, _, K_zs_val = self.get_all_kernels(zl_array, zs_array)
            return K_zs_val
    
    def time_delay(self, y, sigma, zl, zs):
        """
        计算时间延迟
        
        参数:
        y: 像位置参数
        sigma: 速度弥散 (km/s)
        zl: 透镜红移
        zs: 源红移
        
        返回:
        时间延迟 (hrs)
        """
        c_km = self.c_km
        Dl = self.cosmo._comoving_distance(zl)
        Ds = self.cosmo._comoving_distance(zs)
        Dls = Ds - Dl
        
        # 时间延迟公式 (小时)
        delay = 32 * np.pi**2 * y * (sigma / c_km)**4 / c_km / 3600 * Dls * Dl / Ds
        
        return delay
    
    def ps(self, zs_array):
        """源红移分布函数"""
        #return np.interp(zs_array, self.zsa, self.psa)
        #先用一个高斯分布做测试
        
        sigma = 3
        mu = 2
        coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((zs_array - mu) / sigma) ** 2
        return coefficient * np.exp(exponent)
        
    
    def ptau(self, sigma_array, zl_array, zs):
        """
        向量化版本的微分光学深度计算
        
        参数:
        sigma_array: 速度弥散数组 (km/s)
        zl_array: 透镜红移数组
        zs: 源红移
        
        返回:
        dtau数组
        """
        # 从类参数中提取值
        fpbh = self.parameters['fpbh']
        Mpbh = self.parameters['Mpbh']
        xcl = self.parameters['xcl']
        xi0 = self.parameters['xi0']
        cosmology_type = self.parameters['cosmology_type']
        model = self.parameters['model']
        mode = self.parameters['mode']
        
        # 质量范围限制
        mass_min = self.mmin #Msun/h
        mass_max = self.mmax
        
        # 计算对应的质量数组
        ml_array = np.array([sis_velocity_dispersion(zl, None, sigma) 
                           for zl, sigma in zip(zl_array, sigma_array)])
        
        # 创建有效掩码
        valid_mask = (ml_array >= mass_min) & (ml_array <= mass_max)
        
        # 初始化结果数组
        dtau_array = np.zeros_like(sigma_array, dtype=float)
        
        if not np.any(valid_mask):
            return dtau_array
        
        # 提取有效数据
        valid_sigma = sigma_array[valid_mask]
        valid_zl = zl_array[valid_mask]
        valid_ml = ml_array[valid_mask]
        
        
        try:
            # 向量化计算质量函数
            dndlnM_array = halo_mass_function(valid_ml, valid_zl,
                                            cosmology_type=cosmology_type,
                                            model=model,
                                            fpbh=fpbh,
                                            Mpbh=Mpbh,
                                            xcl=xcl,
                                            xi0=xi0,
                                            mode=mode)
        except Exception as e:
            if "too large" in str(e) or "R =" in str(e):
                return dtau_array
            else:
                raise e
        
        # 向量化计算后续步骤
        dndv_array = dndlnM_array * 3 / valid_sigma
        
        #dndv_array = test_halo_mass_function_v(valid_sigma, valid_zl)
        sig_c_array = valid_sigma / self.c_km
        
        # 预计算宇宙学量
        Hz_l_array = self.cosmo.Hz(valid_zl) #1/s
        Dl_array = self.cosmo._comoving_distance(valid_zl)
        Ds = self.cosmo._comoving_distance(zs)
        Dls_array = Ds - Dl_array
        
        # 向量化计算dtau
        dtau_valid = (16 * np.pi**3 * self.c_km / Hz_l_array * 
                     Dls_array**2 * Dl_array**2 / Ds**2 * 
                     dndv_array * sig_c_array**4 / self.Mpc_h_km**3)
        
        # 处理无效值
        dtau_valid = np.nan_to_num(dtau_valid, nan=0.0, posinf=0.0, neginf=0.0)
        dtau_valid[dtau_valid < 0] = 0.0
        
        # 将有效结果赋值回原数组
        dtau_array[valid_mask] = dtau_valid
        
        return dtau_array
    
    def integrate_tau(self, zs_array):
        """
        快速光学深度积分（使用预计算的插值函数）
        
        参数:
        zs: 源红移
        
        返回:
        光学深度 tau
        """
        # 检查是否需要重新预计算
        if self.tau_interpolator is None:
            print("未预计算，进行预计算...")
            self._precompute_tau_integral()
        
        # 使用预计算的插值函数
        return self.tau_interpolator(zs_array)
    
    def g(self, sigma_array, zl_array, zs_array):
        """
        快速版本的概率分布函数（使用预计算的归一化常数）
        
        参数:
        sigma_array: 速度弥散数组 (km/s)
        zl_array: 透镜红移数组
        zs: 源红移
        
        返回:
        概率值数组
        """
        # 从类参数中提取值
        fpbh = self.parameters['fpbh']
        Mpbh = self.parameters['Mpbh']
        xcl = self.parameters['xcl']
        xi0 = self.parameters['xi0']
        cosmology_type = self.parameters['cosmology_type']
        model = self.parameters['model']
        mode = self.parameters['mode']
        
        # 质量范围限制
        mass_min = self.mmin #Msun/h
        mass_max = self.mmax
        
        # 计算对应的质量数组
        ml_array = np.array([sis_velocity_dispersion(zl, None, sigma) 
                           for zl, sigma in zip(zl_array, sigma_array)])
        
        # 创建有效掩码
        valid_mask = (ml_array >= mass_min) & (ml_array <= mass_max)
        
        # 初始化结果数组
        dtau_array = np.zeros_like(sigma_array, dtype=float)
        
        if not np.any(valid_mask):
            return dtau_array
        
        # 提取有效数据
        valid_sigma = sigma_array[valid_mask]
        valid_zl = zl_array[valid_mask]
        valid_ml = ml_array[valid_mask]
        valid_zs = zs_array[valid_mask]
        
        
        try:
            # 向量化计算质量函数
            dndlnM_array = halo_mass_function(valid_ml, valid_zl,
                                            cosmology_type=cosmology_type,
                                            model=model,
                                            fpbh=fpbh,
                                            Mpbh=Mpbh,
                                            xcl=xcl,
                                            xi0=xi0,
                                            mode=mode)
        except Exception as e:
            if "too large" in str(e) or "R =" in str(e):
                return dtau_array
            else:
                raise e
        
        # 向量化计算后续步骤
        dndv_array = dndlnM_array * 3 / valid_sigma #dn/dsigma
        
        #dndv_array = test_halo_mass_function_v(valid_sigma, valid_zl)
        sig_c_array = valid_sigma / self.c_km
        
        # 预计算宇宙学量
        Hz_l_array = self.cosmo.Hz(valid_zl) 
        Dl_array = self.cosmo._comoving_distance(valid_zl)
        Ds_array = self.cosmo._comoving_distance(valid_zs)
        Dls_array = Ds_array - Dl_array
        
        # 向量化计算dtau
        dtau_valid = (16 * np.pi**3 * self.c_km / Hz_l_array * 
                     Dls_array**2 * Dl_array**2 / Ds_array**2 * 
                     dndv_array * sig_c_array**4 / self.Mpc_h_km**3)
        
        # 处理无效值
        dtau_valid = np.nan_to_num(dtau_valid, nan=0.0, posinf=0.0, neginf=0.0)
        dtau_valid[dtau_valid < 0] = 0.0
        
        # 将有效结果赋值回原数组
        dtau_array[valid_mask] = dtau_valid
        
        # 使用预计算的归一化常数
        norm_const = self.integrate_tau(zs_array)
            
        ptau_values = dtau_array / norm_const
        ps_value = self.ps(zs_array)
        
        return ps_value * ptau_values
    
    def compute_N_lens(self, R, T_obs, n_points=200):
        """
        计算透镜时间数量
        
        参数:
        R: 事件量发生率
        T_obs: 观测时长
        n_points: zs的分段数量
        
        返回:
        透镜事件数
        """
        zs_grid = np.linspace(self.zmin, self.zmax, n_points)
        tau_zs = self.integrate_tau(zs_grid)
        dN_array = tau_zs*self.ps(zs_grid)
        
        return np.trapz(dN_array,zs_grid)
    
    def update_parameters(self, precompute_tau=True, **new_kwargs):
        """
        更新参数并可选重新预计算
        
        参数:
        precompute_tau: 是否重新预计算光学深度积分
        **new_kwargs: 新的参数
        """
        print("更新参数...")
        
        # 更新参数
        self.parameters.update(new_kwargs)
        
        # 重新预计算
        if precompute_tau:
            self._precompute_tau_integral()
    
    def get_parameters(self):
        """获取当前参数"""
        return self.parameters.copy()



'''
sigma = np.array([100,120,300])
zl = np.array([0.1, 0.5, 1])
zs = np.array([0.5, 1, 1.5])
# 创建分析对象
analysis = LensingAnalysis(cosmo)

# 使用g函数 - 自动使用相同的参数
result = analysis.time_delay(1, sigma, zl, zs)

y1 = analysis.kernel.K(zl,zs)
y2 = sis_velocity_dispersion(0,None,2000)
print(result, sigma**4*y1, y2)

# 更新参数
analysis.update_parameters(fpbh=1e-2)  # 自动重新预计算

# 再次使用g函数 - 自动使用更新后的参数
result2 = analysis.g(sigma, zl, zs)
'''
