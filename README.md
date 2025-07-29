## [English](README_en.md) | 中文

# PyTorch Transformer 实现

简洁的 PyTorch Transformer 实现，包括模型、训练、分词器和注意力可视化。

### 文件结构

```
train.py                训练脚本
model.py                Transformer 模型定义
config.py               配置参数
dataset.py              数据集读取与预处理
attention_visual.ipynb  注意力可视化（Notebook）
tokenizer_en.json       英文分词器
tokenizer_it.json       意大利文分词器
dataset/                示例数据集
runs/                   TensorBoard 日志
weights/                模型权重保存
```

### 推荐环境

- python == 3.11.13  
- torch == 2.7.1+cu118

*请确保已正确安装 CUDA 驱动以支持 GPU 加速。*

### 使用说明

1. **训练模型：**  
   ```bash
   python train.py
   ```
2. **监测训练过程：**  
   ```bash
   tensorboard --logdir=runs
   ```
3. **注意力可视化：**  
   在 Jupyter Notebook 中打开 `attention_visual.ipynb`。

### 代码参考

本仓库部分代码参考自：  
https://www.youtube.com/watch?v=ISNdQcPhsts&t=50s

# autodl-tmp-code-pytorch-transformer-master
