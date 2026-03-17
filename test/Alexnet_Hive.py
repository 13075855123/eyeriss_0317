import sys
from pathlib import Path
import os
import numpy as np
import skimage.io as io
from skimage.transform import resize
import torch

# 将项目根目录加入环境变量，确保能够以绝对或相对路径运行
project_root = Path(__file__).resolve().parents[1]
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.Hive import Hive
from src.IO2 import RLE
from src.EyerissF import EyerissF as EF

# 如果您按照之前的步骤创建了 model/alexnet.py，可以在此导入用于对比验证
# from model.alexnet import AlexNet
# net = AlexNet()
# net.eval()

# 配置图片所在的路径 (请将其替换为您存放测试图片的实际路径)
# AlexNet 需要 224x224 的 RGB 图片
dir_name = str(project_root / 'imagenet_samples')
if not os.path.exists(dir_name):
    print(f"Warning: Directory {dir_name} does not exist. Please create it and add some test images.")
    files = []
else:
    files = os.listdir(dir_name)

# 由于 AlexNet 计算量极大且特征图很大，仿真时 batch_size 建议设为 1
batch_size = 1
r = RLE(1)
ef = EF()
hive = Hive(ef)

for f in range(0, len(files), batch_size):
    pics = []
    for i in range(batch_size):
        if f + i >= len(files):
            break
        load_from = os.path.join(dir_name, files[f+i])
        # 读取 RGB 图片
        image = io.imread(load_from)
        # 将图片 Resize 到 AlexNet 的标准输入大小 224x224
        image = resize(image, (224, 224), anti_aliasing=True)
        # 转换为 PyTorch 和仿真器通常所需的 (Channel, Height, Width) 格式
        # 即从 (224, 224, 3) 变为 (3, 224, 224)
        if len(image.shape) == 3:
            pic = np.transpose(image, (2, 0, 1))
        else:
            # 如果读入的是灰度图，将其复制成三通道
            pic = np.stack((image,)*3, axis=0)
        pics.append(pic)
        
    pics = np.array(pics, dtype=np.float32)
    inputs = torch.tensor(pics, dtype=torch.float32)
    print(f"Input shape: {pics.shape}")
    
    # ==========================================
    # Layer 1: Conv1 + ReLU + Pool1
    # ==========================================
    print("Running Layer 1...")
    flts = np.load(str(project_root / 'filter' / 'features.c1.weight.npy'))
    pics_compressed = r.Compress(pics)
    flts_compressed = r.Compress(flts)
    
    # 注意: AlexNet 第一层卷积 kernel=11, stride=4, padding=2
    # 提示: 您必须修改 hive.Conv2d 以支持 stride 和 padding 参数！
    pics_out = hive.Conv2d(pics_compressed, flts_compressed, stride=4, padding=2)
    pics_out = hive.PreProcess(pics_out)
    pics_out = hive.ReLU(pics_out)
    
    # AlexNet 第一层池化 kernel=3, stride=2
    pics_out = hive.Pooling(pics_out, kernel_size=3, stride=2)
    
    # ==========================================
    # Layer 2: Conv2 + ReLU + Pool2
    # ==========================================
    print("Running Layer 2...")
    flts = np.float16(np.load(str(project_root / 'filter' / 'features.c3.weight.npy')))
    flts_compressed = r.Compress(flts)
    pics_compressed = r.Compress(pics_out)
    
    # Conv2 kernel=5, stride=1, padding=2
    pics_out = hive.Conv2d(pics_compressed, flts_compressed, stride=1, padding=2)
    pics_out = hive.PreProcess(pics_out)
    pics_out = hive.ReLU(pics_out)
    
    # Pool2 kernel=3, stride=2
    pics_out = hive.Pooling(pics_out, kernel_size=3, stride=2)

    # ==========================================
    # Layer 3: Conv3 + ReLU (No Pooling)
    # ==========================================
    print("Running Layer 3...")
    flts = np.float16(np.load(str(project_root / 'filter' / 'features.c5.weight.npy')))
    flts_compressed = r.Compress(flts)
    pics_compressed = r.Compress(pics_out)
    
    # Conv3 kernel=3, stride=1, padding=1
    pics_out = hive.Conv2d(pics_compressed, flts_compressed, stride=1, padding=1)
    pics_out = hive.PreProcess(pics_out)
    pics_out = hive.ReLU(pics_out)

    # ==========================================
    # Layer 4: Conv4 + ReLU (No Pooling)
    # ==========================================
    print("Running Layer 4...")
    flts = np.float16(np.load(str(project_root / 'filter' / 'features.c6.weight.npy')))
    flts_compressed = r.Compress(flts)
    pics_compressed = r.Compress(pics_out)
    
    # Conv4 kernel=3, stride=1, padding=1
    pics_out = hive.Conv2d(pics_compressed, flts_compressed, stride=1, padding=1)
    pics_out = hive.PreProcess(pics_out)
    pics_out = hive.ReLU(pics_out)

    # ==========================================
    # Layer 5: Conv5 + ReLU + Pool5
    # ==========================================
    print("Running Layer 5...")
    flts = np.float16(np.load(str(project_root / 'filter' / 'features.c7.weight.npy')))
    flts_compressed = r.Compress(flts)
    pics_compressed = r.Compress(pics_out)
    
    # Conv5 kernel=3, stride=1, padding=1
    pics_out = hive.Conv2d(pics_compressed, flts_compressed, stride=1, padding=1)
    pics_out = hive.PreProcess(pics_out)
    pics_out = hive.ReLU(pics_out)
    
    # Pool5 kernel=3, stride=2
    pics_out = hive.Pooling(pics_out, kernel_size=3, stride=2)
    print(f"Shape after Conv layers: {pics_out.shape}") # 预期应该是 (batch_size, 256, 6, 6)

    # ==========================================
    # Fully Connected Layers
    # ==========================================
    print("Running Fully Connected Layers...")
    
    # 将特征图展平为向量
    vector = pics_out.reshape(batch_size, -1)
    
    # FC1 (对应 f9) - 输入 256*6*6=9216，输出 4096
    flts_fc1 = np.load(str(project_root / 'filter' / 'classifier.f9.weight.npy'))
    vector = hive.FullConnect(vector, flts_fc1)
    vector = hive.ReLU(vector)
    
    # FC2 (对应 f10) - 输入 4096，输出 4096
    flts_fc2 = np.load(str(project_root / 'filter' / 'classifier.f10.weight.npy'))
    vector = hive.FullConnect(vector, flts_fc2)
    vector = hive.ReLU(vector)
    
    # FC3 (对应 f11) - 输入 4096，输出分类数 (如 ImageNet 1000)
    flts_fc3 = np.load(str(project_root / 'filter' / 'classifier.f11.weight.npy'))
    vector = hive.FullConnect(vector, flts_fc3)
    
    print(f"Final output shape: {vector.shape}")
    print("Predicted class index : ", vector.argmax(axis=1))
    
    # 为了演示目的，只运行第一个 batch 后就结束。如果您要测试整个文件夹请注释掉 break
    break