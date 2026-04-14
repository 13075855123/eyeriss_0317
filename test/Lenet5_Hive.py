import sys
import os
from pathlib import Path

# ==========================================
# 1. 环境与路径配置
# ==========================================
project_root = Path(__file__).resolve().parents[1]
project_root_str = str(project_root)  
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

import numpy as np
import skimage.io as io
import torch
from model.lenet import LeNet5
from src.Hive import Hive
from src.IO2 import RLE
from src.EyerissF import EyerissF as EF

# ==========================================
# 2. 辅助函数：打印每层卷积的 Pass 切分信息
# ==========================================
def print_pass_info(layer_name, hive, input_data, weights, stride=1):
    """
    在将数据投入硬件模拟前，预览并打印调度器 (Hive/GLB) 的任务切分状态
    """
    hive.stride = stride
    passes = hive.CreatePasses(input_data, weights)
    
    print(f"\n[{layer_name}] 硬件映射与任务切分信息:")
    print(f"  -> 总计生成 Pass 任务数: {len(passes)} 个")
    print(f"  -> PE 阵列映射参数: t(Filter并行数)={hive.t}, r(Channel并行数)={hive.r}, e(行切片)={hive.e}")
    
    if len(passes) > 0:
        pic_pass, weight_pass = passes[0]
        print(f"  -> 首个 Pass 送入 PE 的数据尺寸: 输入块 {pic_pass.shape}, 权重块 {weight_pass.shape}")

# ==========================================
# 3. 主程序
# ==========================================
def main():
    # --- 加载 PyTorch LeNet5 标准模型与权重 ---
    net = LeNet5()
    print(">>> 正在加载网络权重...")
    for i, (name, param) in enumerate(net.named_parameters()):
        data = np.load(str(project_root / 'filter' / f'{name}.npy'))
        param.data = torch.from_numpy(data)
    net.eval()

    # --- 数据集加载配置 ---
    dir_name = str(project_root / 'mnist_png' / 'mnist_png' / 'training' / '5')
    files = sorted(os.listdir(dir_name))[:4] # 测试前4张图片
    batch_size = 4

    # --- 初始化 Eyeriss 硬件模拟器 ---
    r = RLE(1)          # 游程编码压缩器
    ef = EF()           # 12x14 PE阵列
    hive = Hive(ef)     # 全局调度器 (GLB)

    # ==========================================
    # 4. 执行推理 (按批次)
    # ==========================================
    for f in range(0, len(files), batch_size):
        # 动态计算当前批次实际包含几张图（防止最后一次越界）
        current_batch_size = min(batch_size, len(files) - f)
        
        # 1. 图像读取与预处理
        pics = []
        for i in range(current_batch_size):
            load_from = os.path.join(dir_name, files[f+i])
            image = io.imread(load_from, as_gray=True)
            image = np.pad(image, ((2,2),(2,2)), 'median') # 使用 np.pad 消除警告
            pic = np.array(image/255.0).reshape(1, image.shape[0], -1)
            pics.append(pic) 
            
        x = np.array(pics) # x 的形状: (Batch, Channel=1, H=32, W=32)
        inputs = torch.tensor(x, dtype=torch.float32) # 留给 PyTorch 验证用
        
        # ---------------------------------------------------------
        # 第一层卷积 (Convnet C1)
        # ---------------------------------------------------------
        flts_c1 = np.load(str(project_root / 'filter' / 'convnet.c1.weight.npy'))
        print_pass_info("C1 Convolution", hive, x, flts_c1, stride=1)
        
        x_comp = r.Compress(x)
        flts_comp = r.Compress(flts_c1)
        x = hive.Conv2d(x_comp, flts_comp, stride=1)
        
        x = hive.PreProcess(x) # RLE 解压
        x = hive.ReLU(x)
        x = hive.Pooling(x, kernel_size=2, stride=2) # 计算机(CPU)执行池化
        
        # ---------------------------------------------------------
        # 第三层卷积 (Convnet C3)
        # ---------------------------------------------------------
        flts_c3 = np.float16(np.load(str(project_root / 'filter' / 'convnet.c3.weight.npy')))
        print_pass_info("C3 Convolution", hive, x, flts_c3, stride=1)
        
        x_comp = r.Compress(x)
        flts_comp = r.Compress(flts_c3)
        x = hive.Conv2d(x_comp, flts_comp, stride=1)
        
        x = hive.PreProcess(x)
        x = hive.ReLU(x)
        x = hive.Pooling(x, kernel_size=2, stride=2)
        
        # ---------------------------------------------------------
        # 第五层卷积 (Convnet C5)
        # ---------------------------------------------------------
        flts_c5 = np.float16(np.load(str(project_root / 'filter' / 'convnet.c5.weight.npy')))
        print_pass_info("C5 Convolution", hive, x, flts_c5, stride=1)
        
        x_comp = r.Compress(x)
        flts_comp = r.Compress(flts_c5)
        x = hive.Conv2d(x_comp, flts_comp, stride=1)
        
        x = hive.PreProcess(x)
        x = hive.ReLU(x)

        # ---------------------------------------------------------
        # 精度验证 (与 PyTorch 标准实现对比)
        # ---------------------------------------------------------
        res = inputs
        for i in range(8): # 前8层刚好是卷积和池化部分
            res = net.convnet[i](res)
            
        diff = x - res.data.numpy()
        max_err = np.max(np.abs(diff))
        print(f"\n[Accuracy] C5 输出与 PyTorch 对比, 最大绝对误差: {max_err:.6f}")

        # ---------------------------------------------------------
        # 全连接层 (FC6, FC7)
        # ---------------------------------------------------------
        # 将张量拉平，注意使用真实的 current_batch_size
        vector = x.reshape(current_batch_size, -1)
        
        fc6_weight = np.load(str(project_root / 'filter' / 'fc.f6.weight.npy'))
        vector = hive.FullConnect(vector, fc6_weight)
        vector = hive.ReLU(vector)
        
        fc7_weight = np.load(str(project_root / 'filter' / 'fc.f7.weight.npy'))
        vector = hive.FullConnect(vector, fc7_weight)

        print(f"\n[Result] 最终预测的数字结果: {vector.argmax(axis=1)}")

if __name__ == "__main__":
    main()