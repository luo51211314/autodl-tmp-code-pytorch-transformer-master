## English | [中文](README.md)

# PyTorch Transformer Implementation

A concise PyTorch implementation of Transformer, including model, training, tokenizers, and attention visualization.

### File Structure

```
train.py                Model training script
model.py                Transformer model definition
config.py               Configuration parameters
dataset.py              Dataset reading and preprocessing
attention_visual.ipynb  Attention visualization (Jupyter Notebook)
tokenizer_en.json       English tokenizer
tokenizer_it.json       Italian tokenizer
dataset/                Sample datasets
runs/                   TensorBoard logs
weights/                Saved model weights
```

### Recommended Environment

- python == 3.11.13  
- torch == 2.7.1+cu118

*Please ensure compatible CUDA drivers for GPU acceleration.*

### Usage

1. **Train the model:**  
   ```bash
   python train.py
   ```
2. **Monitor training:**  
   ```bash
   tensorboard --logdir=runs
   ```
3. **Visualize attention mechanism:**  
   Open `attention_visual.ipynb` in Jupyter Notebook.

### Reference

The code is based on the tutorial:  
https://www.youtube.com/watch?v=ISNdQcPhsts&t=50s

