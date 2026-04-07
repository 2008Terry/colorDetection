# White Line Detection (Soccer Field) - V2

这个版本针对复杂画面（鱼眼、反光、阴影、低对比）做了增强，支持亮线/暗线/双极性一起检测。

## 依赖

```bash
pip install opencv-python numpy
```

## 用法

```bash
python detect_white_lines.py --input your_image.jpg --outdir outputs
```

### 常用参数

- `--line-polarity {bright,dark,both}`
  - `bright`：只检测亮白线
  - `dark`：只检测暗线（有些图里白线会看起来偏暗）
  - `both`：同时检测（默认，最稳妥）
- `--min-line-length`：Hough 最短线段（默认 45）
- `--max-line-gap`：线段连接间隔（默认 14）
- `--debug`：输出额外调试图

## 输出

- `01_field_mask.png`：场地区域掩膜
- `02_enhanced.png`：线结构增强图
- `03_candidates.png`：候选线像素
- `04_pruned.png`：连通域筛选后结果
- `05_hough_lines.png`：Hough 线段结果
- `06_final_mask.png`：最终边线掩膜
- `07_overlay.png`：叠加可视化
