# R0101 回归训练脚本

用于读取 `R0101.csv`，选择 `PRES`、`ATM`、`TEMP` 等列作为输入特征，`MASSFRA` 作为标签，训练多种回归模型并生成可视化结果。

快速开始：

1. 创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 运行脚本：

```bash
python train_regressions.py --csv R0101.csv --out outputs
```

输出文件会保存到 `outputs` 目录，包含每个模型的预测图、残差图和 `results_summary.csv`。

注意：脚本会尝试处理多行表头（若CSV文件在前几行包含层级列名）。如果未能自动识别列名，请手动预处理CSV以包含 `MASSFRA` 列名和所需特征列。
