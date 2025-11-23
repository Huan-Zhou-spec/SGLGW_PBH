用于计算加入超大质量原初黑洞造成的物质功率谱、暗物质晕质量分布、引力透镜时间延迟等
ps_pbh/
└── modules               # 所有模块集合
    ├── constants.py      # 常用的物理学、宇宙学常数
    ├── Cosmo.py          # 宇宙学模拟器
    ├── Function.py       # 物质功率谱与暗物质晕质量分布函数
    ├── Lensis.py         # sis模型下速度弥散、时间延迟、光深计算
    ├── NIM.py            # 时间延迟分布数值计算器
    ├── mcmc.py           # 参数估计模块
    ├── interpolators.py  # 插值模块
                
└── PsHmfPlots.py         # 功率谱与质量分布画图
└── fpbhData.py           # 用于生成不同模型下的fpbh参数数据集
└── fpbhPlots.py          # 时间延迟分布画图
└── FiducialPlots.py      # 用于生成基准模型时间延迟数据与分布图片
└── McmcPlots.py          # 用于计算原初黑洞丰度后验分布的图片
    
└── data                  # 所有输入和输出数据集合 

└── Plots                 # 所有输出图集合
 
└── test                  # 存放一些测试文件，暂时无用
持续更新中
版本1.0.0
