# 项目对比：qtsnn vs quantile-regression-master

## 文件结构对比

### quantile-regression-master (原始项目)
```
quantile-regression-master/
└── python/
    ├── benchmark.py          # 原始benchmark脚本
    ├── forest_model.py       # 随机森林模型
    ├── funcs.py              # 原始数据生成场景
    ├── neural_model.py       # 神经网络分位数回归
    ├── neural_sqerr.py       # 平方误差神经网络
    ├── spline_model.py       # 样条模型
    ├── torch_utils.py        # PyTorch工具
    ├── utils.py              # 通用工具
    └── visualize.py          # 可视化函数
```

### qtsnn (新项目 - 时空数据版本)
```
qtsnn/
├── benchmark.py                      # 原始benchmark (保留)
├── benchmark_st.py                   # 新增：时空数据benchmark ⭐
├── spatial_temporal_scenarios.py    # 新增：时空数据生成场景 ⭐
├── forest_model.py                   # 复制自原项目
├── funcs.py                          # 复制自原项目
├── neural_model.py                   # 复制自原项目
├── neural_sqerr.py                   # 复制自原项目
├── spline_model.py                   # 复制自原项目
├── torch_utils.py                    # 复制自原项目
├── utils.py                          # 复制自原项目
├── visualize.py                      # 复制自原项目
├── requirements.txt                  # 新增：依赖管理 ⭐
├── README.md                         # 新增：中文文档 ⭐
└── [data/, plots/, results/]         # 输出目录
```

## 核心差异

### 1. 完全相同的文件（直接复制）
以下文件与原项目**完全一致**：
- `neural_model.py` - 神经网络分位数回归模型
- `neural_sqerr.py` - 平方误差基线模型
- `utils.py` - 工具函数
- `torch_utils.py` - PyTorch工具
- `visualize.py` - 可视化函数
- `forest_model.py` - 随机森林模型
- `spline_model.py` - 样条模型
- `funcs.py` - 原始数据场景
- `benchmark.py` - 原始benchmark脚本

### 2. 新增文件（qtsnn独有）

#### ⭐ `spatial_temporal_scenarios.py`
**作用**：从QTS-main项目的R代码转换而来的时空数据生成场景

**包含3个场景**：
- `STScenario1`: 分段常数函数 + 正弦基空间依赖
- `STScenario2`: 线性边界函数 + t分布误差
- `STScenario3`: 分位数特定的beta + t(3)误差

**关键特性**：
- 时间依赖：协变量和误差的AR(1)过程
- 空间依赖：通过正弦基函数建模
- 变化的观测数量：每个时间点观测数不同

#### ⭐ `benchmark_st.py`
**作用**：专门用于时空数据的benchmark脚本

**与原始benchmark.py的区别**：
| 特性 | benchmark.py (原始) | benchmark_st.py (新) |
|------|-------------------|---------------------|
| 数据场景 | Scenario1-5 (非时空) | STScenario1-3 (时空) |
| 样本量 | [100, 1000, 10000] | [1000, 5000, 10000] |
| 模型数量 | 4个 (NN, Spline, Forest, SqErr) | 2个 (NN, SqErr) |
| 命令行参数 | 无 | --demo, --full ⭐ |
| 输出目录 | data/ | results/ |
| 可视化 | 所有场景 | 仅Scenario1 (demo模式) |

**新增功能**：
```python
# 支持命令行参数
python benchmark_st.py          # demo模式（默认）
python benchmark_st.py --demo   # 显式demo模式
python benchmark_st.py --full   # 完整实验（100次试验）
```

#### ⭐ `requirements.txt`
```
numpy
scipy
torch
matplotlib
seaborn
scikit-learn
```

#### ⭐ `README.md`
- 中文文档
- 详细的运行说明
- 场景描述
- 参数配置指南

## 使用场景对比

### 原项目 (quantile-regression-master)
**适用于**：
- 一般的分位数回归问题
- 非时空数据
- 比较多种模型（NN, Spline, Forest）
- 研究论文复现

**运行方式**：
```bash
cd quantile-regression-master/python
python benchmark.py
```

### 新项目 (qtsnn)
**适用于**：
- 时空数据的分位数回归
- 具有时间和空间依赖性的数据
- 专注于神经网络方法
- 快速原型和实验

**运行方式**：
```bash
cd qtsnn
python benchmark_st.py --demo   # 快速测试
python benchmark_st.py --full   # 完整实验
```

## 数据生成对比

### 原项目场景 (funcs.py)
```python
# 示例：Scenario1
X ~ Uniform[0,1]^d
y = f(X) + ε
ε ~ N(0, σ²)
# 无时空依赖
```

### 新项目场景 (spatial_temporal_scenarios.py)
```python
# 示例：STScenario1
# 时间依赖
X[t] = 0.1 * X[t-1] + 0.9 * U[0,1]

# 空间依赖
spatial_error[t] = Σ δ_k * sin(2πk·X)

# 时间相关误差
ε[t] = ρ * ε[t-1] + √(1-ρ²) * η[t]

# 最终响应
y[t] = f(X[t]) + spatial_error[t] + ε[t]
```

## 模型架构对比

### 完全相同
两个项目使用**完全相同**的神经网络架构：
- 输入层 → 200 (Dropout + ReLU + BatchNorm)
- 隐藏层 → 200 (Dropout + ReLU + BatchNorm)
- 输出层 → n_quantiles
- 优化器：SGD with Nesterov momentum
- 分位数单调性约束：通过softplus强制

## 总结

### 核心模型代码：100%相同
- `neural_model.py`
- `neural_sqerr.py`
- `utils.py`
- `torch_utils.py`
- `visualize.py`

### 主要区别：数据生成和benchmark脚本
1. **新增时空数据场景** (`spatial_temporal_scenarios.py`)
2. **新增时空benchmark** (`benchmark_st.py`)
3. **改进的用户体验**（命令行参数、中文文档）

### 兼容性
- qtsnn项目**完全保留**了原项目的所有功能
- 可以同时运行原始benchmark和时空benchmark
- 模型训练和预测逻辑完全一致

### 建议
- 如果研究**一般分位数回归**：使用 `benchmark.py`
- 如果研究**时空数据**：使用 `benchmark_st.py`
- 两者可以在同一项目中共存
