import json
import numpy as np
import torch

# 自定义 JSONEncoder 以处理所有可能的 numpy 和 Tensor 数据类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # 处理 ndarray 类型
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # 将 NumPy 数组转换为列表
        
        # 处理 numpy 的各种数值类型
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)  # 将 numpy 的整数类型转换为普通的 Python int
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)  # 将 numpy 的浮动类型转换为 Python float
        
        # 处理 Tensor 类型（假设你使用的是 PyTorch）
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()  # 将 PyTorch tensor 转换为 numpy 数组再转换为列表

        # 如果是 TensorFlow 的 Tensor
        try:
            import tensorflow as tf
            if isinstance(obj, tf.Tensor):
                return obj.numpy().tolist()  # 将 TensorFlow tensor 转换为 numpy 数组再转换为列表
        except ImportError:
            pass  # 如果没有安装 tensorflow，可以忽略

        return super().default(obj)  # 调用父类方法处理其他对象