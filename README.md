# ELM Radiomics Pipeline

这是一个基于极限学习机 (Extreme Learning Machine, ELM) 的影像组学分析流程。该项目将原始的 MATLAB/Python 混合代码重构为纯 Python 实现，提供了从特征提取、筛选、模型训练到评估的完整流水线。

## 目录

- [项目结构](#项目结构)
- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [详细步骤说明](#详细步骤说明)
- [单独运行步骤](#单独运行步骤)

## 项目结构

```
elm/
├── config_default.yaml      # 默认配置文件
├── organize_data.py         # 数据整理脚本
├── run_pipeline.py          # 主程序，一键运行全流程
├── step1_feature_extraction.py  # 步骤1：特征提取与合并
├── step2_normalization.py       # 步骤2：Z-score 归一化
├── step3_utest_selection.py     # 步骤3：U-test 特征初筛
├── step4_mrmr_selection.py      # 步骤4：MRMR 特征精选 (支持多线程)
├── step5_elm_training.py        # 步骤5：ELM 模型训练与交叉验证
├── step6_evaluation.py          # 步骤6：模型评估指标计算
├── step7_roc_analysis.py        # 步骤7：ROC 曲线绘制与分析
├── utils.py                     # 通用工具函数
└── ORGANIZE_DATA_README.md      # 数据整理说明文档
```

## 环境安装

需要 Python 3.8+。建议创建虚拟环境：

```bash
# 安装依赖
pip install numpy scipy scikit-learn matplotlib pandas pyyaml
```

## 数据准备

原始数据应按照以下结构存放（支持嵌套目录）：

```
data/
└── {label}/                 # 标签目录 (如 0, 1)
    └── {patient_id}/        # 患者ID目录
        └── {patient_id}_{modality}.mat  # 特征文件 (如 123_A.mat)
```

**第一步：整理数据**

使用 `organize_data.py` 将原始数据整理为流水线所需的格式：

```bash
# 整理数据，生成 train/test/test1 划分
python organize_data.py --data_root data --output_dir organized_data --modalities A P
```

这将生成 `organized_data` 目录，包含 `jiangmen_A_CMTA` 和 `jiangmen_P_CMTA` 等子目录，以及对应的日志文件 (`train_A_log.txt` 等)。

## 快速开始

整理好数据后，使用 `run_pipeline.py` 运行完整流程：

```bash
# 运行所有模态 (A, P)
python run_pipeline.py --data_root organized_data

# 只运行特定模态，并指定使用 16 个线程进行 MRMR 计算
python run_pipeline.py --data_root organized_data --data_types A --n_jobs 16
```

## 配置说明

主要参数可以在 `config_default.yaml` 中修改，或者通过命令行参数覆盖：

| 参数 | 命令行参数 | 说明 | 默认值 |
|------|------------|------|--------|
| **U-test** | | | |
| p_threshold | `--p_threshold` | U-test 显著性阈值 | 0.05 |
| **MRMR** | | | |
| n_features | `--n_features` | MRMR 最终选择的特征数 | 50 |
| n_jobs | `--n_jobs` | 并行线程数 (-1 为全核) | -1 |
| **ELM** | | | |
| n_hidden_min | `--n_hidden_min` | 最小隐藏层神经元数 | 2 |
| n_hidden_max | `--n_hidden_max` | 最大隐藏层神经元数 | 5 |
| n_folds | `--n_folds` | 交叉验证折数 | 5 |
| n_trials | `--n_trials` | 随机搜索次数 | 100 |

## 详细步骤说明

1. **特征提取 (Step 1)**: 读取整理后的 `.mat` 文件，计算每个患者的特征均值，合并为特征矩阵。
2. **归一化 (Step 2)**: 对训练集进行 Z-score 归一化，并利用训练集的均值/标准差对测试集进行归一化，防止数据泄露。
3. **U-test 筛选 (Step 3)**: 在训练集上进行 Mann-Whitney U 检验，保留 p 值小于阈值 (0.05) 的显著特征。
4. **MRMR 筛选 (Step 4)**: 使用最小冗余最大相关 (mRMR) 算法从 U-test 筛选后的特征中进一步选择 50 个特征。此步骤支持多线程加速。
5. **ELM 训练 (Step 5)**: 使用 ELM 分类器。通过交叉验证 (CV) 在训练集上寻找最佳的隐藏层神经元数量。为了克服 ELM 的随机性，会进行多次随机试验 (`n_trials`)。
6. **评估 (Step 6)**: 计算训练集、测试集、外部测试集 (test1) 的详细指标：AUC, Sensitivity, Specificity, Accuracy, PPV, NPV, Confusion Matrix。
7. **ROC 分析 (Step 7)**: 绘制 ROC 曲线，计算 AUC 及其 95% 置信区间，并保存图表。

## 单独运行步骤

每个步骤的脚本都可以独立运行。例如，如果你调整了 MRMR 参数，只想重新运行特征选择及之后的步骤：

```bash
# 1. 运行 MRMR
python step4_mrmr_selection.py --input organized_data/jiangmen_A_CMTA/feature_Utest.mat --output organized_data/jiangmen_A_CMTA/MRMRfeature.mat --n_features 50 --n_jobs -1

# 2. 运行 ELM 训练
python step5_elm_training.py --input organized_data/jiangmen_A_CMTA/MRMRfeature.mat --output organized_data/jiangmen_A_CMTA/elm_model_result_A.mat --n_trials 200

# 3. 运行评估
python step6_evaluation.py --input organized_data/jiangmen_A_CMTA/elm_model_result_A.mat --output_dir organized_data/jiangmen_A_CMTA/evaluation

# 4. 运行 ROC 分析
python step7_roc_analysis.py --input organized_data/jiangmen_A_CMTA/elm_model_result_A.mat --output_dir organized_data/jiangmen_A_CMTA/roc_analysis
```

## 输出结果

结果保存在 `organized_data/{dataset_name}/` 目录下：

- `feature_normalized.mat`: 归一化后的特征
- `MRMRfeature.mat`: 最终选定的特征
- `elm_model_result_{type}.mat`: 模型权重及预测结果
- `evaluation/`: 包含各数据集的指标 (.mat) 和汇总 CSV (`metrics_summary.csv`)
- `roc_analysis/`: 包含 ROC 曲线图 (`roc_curves.png`) 和 AUC 数据
