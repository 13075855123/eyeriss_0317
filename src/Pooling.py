import numpy as np

def Pooling(array, kernel_size, stride):
    """
    递归处理高维数组，最终对 2D 特征图进行池化。
    强制要求传入 kernel_size 和 stride，彻底解耦。
    """
    # 如果是多维数组（如 Batch 或 Channel 维度），继续递归解包
    if len(array.shape) > 2:
        return np.array([Pooling(x, kernel_size, stride) for x in array])
    
    # 到了真正的 2D 特征图层面，调用具体的 MAXPooling 实现
    return MAXPooling(array, kernel_size, stride)

def MAXPooling(Array, ksize, stride):
    """
    底层的最大池化实现，支持任意重叠/非重叠池化。
    """
    H, W = Array.shape
    
    # 根据传入的步长和核大小，动态计算输出特征图的尺寸
    out_H = (H - ksize) // stride + 1
    out_W = (W - ksize) // stride + 1
    
    out = np.zeros((out_H, out_W), dtype=Array.dtype)
    
    # 滑动窗口计算
    for i in range(out_H):
        for j in range(out_W):
            # 提取当前窗口
            window = Array[i*stride : i*stride + ksize, 
                           j*stride : j*stride + ksize]
            out[i, j] = np.max(window)
            
    return out

