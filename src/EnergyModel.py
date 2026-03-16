import numpy as np
# 如果有 conf 文件，保留引用；如果没有，可以直接在内部定义
# from . import conf 

class EnergyModel:
    """
    Eyeriss 光电融合架构能耗评估模型
    
    该模型包含两部分：
    1. 电域 (Electrical): 基于 65nm 工艺的 Eyeriss/Horowitz 标准数据。
    2. 光域 (Optical): 基于 PhoNoCMap 的光器件参数，针对单波导蛇形拓扑优化。
    """

    def __init__(self):
        # =========================================================================
        # PART 1: 电域能耗参数 (Electrical Parameters) - 65nm Process
        # 来源: 
        # [1] M. Horowitz, "Computing's energy problem", ISSCC 2014 (基准数据)
        # [2] Y. Chen et al., "Eyeriss: A Spatial Architecture...", ISCA 2016 (具体应用)
        # =========================================================================
        
        # 16-bit 乘加运算 (MAC)
        # 包含乘法(~1pJ)、加法(~0.03pJ)及寄存器/流水线开销。工程估算值。
        self.E_MAC = 1.5          # pJ/op

        # 寄存器文件/便笺存储器 (RF/Scratchpad) 读写 (< 1KB)
        # 本地数据访问，能耗极低。
        self.E_RF = 1.0           # pJ/word (16-bit)

        # 全局缓存 (Global Buffer) 读写 (> 100KB SRAM)
        # 大容量 SRAM，电容大，访问能耗高。Horowitz 数据通常在 5-10pJ 范围。
        self.E_GLB = 10.0         # pJ/word (16-bit)

        # 片外 DRAM 访问 (LPDDR)
        # 相比片上操作高 2-3 个数量级。Eyeriss 论文中 DRAM 约为 MAC 的 200-500 倍。
        self.E_DRAM = 640.0       # pJ/word (16-bit)

        # 电互连传输 (Psum 向上传输)
        # 65nm 下每毫米线驱动功耗。用于计算部分和 (Partial Sum) 的电传输。
        self.E_Elec_Wire = 10.0   # pJ/hop (假设每跳约 1mm)

        # =========================================================================
        # PART 2: 光域能耗参数 (Optical Parameters)
        # 来源: PhoNoCMap User Guide
        # =========================================================================
        
        # --- 静态功耗 (Static Power) ---
        # 只要系统运行就需要持续消耗，单位 mW
        
        # 激光器阈值功耗 (Laser Threshold Power)
        # 激光器必须达到此电流阈值才能发光。
        self.P_Laser_Threshold = 10.0  # mW
        
        # 微环热调谐功耗 (Microring Heating/Tuning)
        # 为了锁定谐振波长，必须对微环加热对抗温度漂移。
        self.P_MRR_Heating = 5.0       # mW/ring

        # --- 动态能耗 (Dynamic Energy) ---
        # 随传输数据量变化，单位 pJ/bit
        
        # 电转光 (E-O Modulation): 驱动调制器消耗的能量
        self.E_EO_Modulation = 0.1     # pJ/bit
        
        # 光转电 (O-E Detection): 探测器+TIA+判决电路的能量
        self.E_OE_Detection = 0.1      # pJ/bit
        
        # 微环开关 (MRR Switching): 动态改变微环状态(Drop/Pass)的能量
        self.E_MRR_Switch = 0.05       # pJ/bit

        # --- 光路物理损耗参数 (Physical Loss Parameters) ---
        # 来源: PhoNoCMap Table 1
        
        self.L_prop_dB_cm = 0.274      # 波导传播损耗 (dB/cm)
        self.L_through_dB = 0.005      # 微环直通损耗 (OFF state) (dB/ring)
        self.L_drop_dB    = 0.5        # 微环下载损耗 (ON state) (dB)
        self.L_margin_dB  = 1.0        # 系统额外裕量 (Coupling loss etc.) (dB)

        # 蛇形结构无交叉损耗 (L_crossing = 0)

        # --- 接收端与激光器效率 ---
        # 探测器灵敏度 (Sensitivity): 保证误码率下的最小接收功率
        self.Sensitivity_dBm = -14.2   # dBm
        # 将 dBm 转换为 mW: P(mW) = 10^(P(dBm)/10)
        self.Sensitivity_mW = 10 ** (self.Sensitivity_dBm / 10.0)

        # 激光器电光转换效率 (Wall-Plug Efficiency)
        self.Laser_Eff = 0.30          # 30%

        # --- 架构尺寸参数 (针对蛇形结构) ---
        self.PE_Rows = 12
        self.PE_Cols = 14
        self.Num_PEs = self.PE_Rows * self.PE_Cols
        self.PE_Pitch_cm = 0.03         # 假设 PE 中心间距为 0.3mm (0.03cm)


    def calculate_laser_power(self, active_wavelengths):
        """
        计算激光器所需的总电功率 (Wall-Plug Power)。
        基于最坏情况损耗 (Worst-Case Loss) 计算。
        
        针对: 单波导蛇形结构 (Single-Waveguide Serpentine)
        最坏路径: 光从源头出发，经过所有 PE，到达最后一个 PE。
        """
        if active_wavelengths == 0:
            return 0.0

        # 1. 传播损耗 (Propagation Loss)
        # 蛇形总长度 = PE数量 * 间距
        total_length_cm = self.Num_PEs * self.PE_Pitch_cm
        loss_prop = total_length_cm * self.L_prop_dB_cm
        
        # 2. 直通损耗 (Through Loss)
        # 最坏情况：光到达最后一个 PE，必须穿过前面所有 PE 的微环。
        # 最后一个 PE 是 Drop (下载)，所以不计入 Through。
        # 每个 PE 有 2 个微环 (Input + Filter)。
        num_through_rings = (self.Num_PEs - 1) * 2
        loss_through = num_through_rings * self.L_through_dB
        
        # 3. 总损耗 (Total Optical Loss)
        # 包含: 传播 + 直通 + 下载(1次) + 裕量
        total_loss_dB = loss_prop + loss_through + self.L_drop_dB + self.L_margin_dB
        
        # 4. 计算到达接收端所需的最小发射光功率 (Required Optical Power at Source)
        # P_source = P_sensitivity * 10^(Loss/10)
        required_optical_power_mW = self.Sensitivity_mW * (10 ** (total_loss_dB / 10.0))
        
        # 5. 计算激光器总耗电功率 (Total Electrical Power)
        # P_elec = P_threshold + (P_optical_total / Efficiency)
        # 假设总光功率 = 单波长所需功率 * 波长数
        total_laser_power_mW = self.P_Laser_Threshold + \
                               (required_optical_power_mW * active_wavelengths / self.Laser_Eff)
        
        return total_laser_power_mW


    def calculate_total_energy(self, stats, time_ns):
        """
        计算一次 Pass (或一个时间段) 的总能耗。
        
        参数:
        stats (dict): 统计数据，包含:
            - 'mac_count': MAC 运算次数
            - 'dram_access_bits': DRAM 访问数据量 (bits)
            - 'glb_access_bits': Global Buffer 访问数据量 (bits)
            - 'rf_access_bits': RF 访问数据量 (bits)
            - 'psum_elec_hops': Partial Sum 电传输的总跳数
            - 'optical_bits': 光互连传输的总数据量 (bits)
            - 'active_wavelengths': 使用的波长数量
        time_ns (float): 该任务运行的总时间 (纳秒)，用于计算静态功耗。
        
        返回:
        total_energy_pJ (float): 总能耗 (皮焦耳)
        breakdown (dict): 各部分能耗明细
        """
        
        # --- A. 电域能耗 (Electrical Energy) ---
        # 计算公式: 次数 * 单次能耗
        
        e_mac = stats['mac_count'] * self.E_MAC
        
        # 存储访问 (注意将 bits 转换为 16-bit word)
        e_dram = (stats['dram_access_bits'] / 16.0) * self.E_DRAM
        e_glb  = (stats['glb_access_bits']  / 16.0) * self.E_GLB
        e_rf   = (stats['rf_access_bits']   / 16.0) * self.E_RF
        
        # Psum 电传输
        e_psum_noc = stats['psum_elec_hops'] * self.E_Elec_Wire

        # --- B. 光域能耗 (Optical Energy) ---
        
        # 1. 动态能耗 (Dynamic): 调制 + 探测 + 开关
        e_opt_dynamic = stats['optical_bits'] * \
                        (self.E_EO_Modulation + self.E_OE_Detection + self.E_MRR_Switch)
        
        # 2. 静态激光器能耗 (Static Laser): 功率 * 时间
        p_laser_mW = self.calculate_laser_power(stats['active_wavelengths'])
        e_laser = p_laser_mW * time_ns  # mW * ns = pJ
        
        # 3. 静态微环加热能耗 (Static Heating): 功率 * 时间
        # 假设所有 PE 的微环都需要热锁定 (Thermal Tuning)
        num_total_rings = self.Num_PEs * 2
        e_heating = (num_total_rings * self.P_MRR_Heating) * time_ns # pJ

        # --- 汇总 ---
        total_energy_pJ = e_mac + e_dram + e_glb + e_rf + e_psum_noc + \
                          e_opt_dynamic + e_laser + e_heating
        
        breakdown = {
            "Electrical_Compute (MAC)": e_mac,
            "Electrical_Memory (DRAM+GLB+RF)": e_dram + e_glb + e_rf,
            "Electrical_NoC (Psum)": e_psum_noc,
            "Optical_Dynamic (Mod/Det)": e_opt_dynamic,
            "Optical_Static_Laser": e_laser,
            "Optical_Static_Heating": e_heating,
            "Total_pJ": total_energy_pJ
        }
        
        return total_energy_pJ, breakdown