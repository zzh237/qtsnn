# Quantile Regression Neural Network for Spatial-Temporal Data

这个项目结合了神经网络分位数回归和时空数据生成，用于在具有时空依赖性的数据上进行分位数回归。

## 项目结构

```
qtsnn/
├── spatial_temporal_scenarios.py  # 时空数据生成场景（从R转换为Python）
├── benchmark_st.py                # 主要运行脚本
├── neural_model.py                # 神经网络分位数回归模型
├── neural_sqerr.py                # 平方误差神经网络基线
├── utils.py                       # 工具函数
├── torch_utils.py                 # PyTorch工具
├── visualize.py                   # 可视化函数
├── logger.py                      # 日志工具
├── plots/                         # 输出图表
├── results/                       # 输出结果
├── logs/                          # 运行日志
└── README.md                      # 本文件
```

## 依赖环境

需要安装以下Python包：

```bash
pip install numpy scipy torch matplotlib seaborn
```

或者使用requirements.txt：

```bash
pip install -r requirements.txt
```

## 数据场景说明

项目包含3个时空数据生成场景（从QTS-main/Multivariate/scenarios.R转换）：

### STScenario1
- 分段常数函数 (d=2)
- 通过正弦基函数的delta函数实现空间依赖
- 空间误差: N(0, 1)，时间依赖系数0.5
- 测量误差: N(0, 0.5)，时间依赖系数0.3

### STScenario2
- 线性边界函数 (d=2)
- 空间误差: t(2)分布，时间依赖系数0.5
- 测量误差: Cauchy(0, 1)，时间依赖系数0.3

### STScenario3
- 分段常数函数，带分位数特定的beta
- 空间误差: N(0, 1)，时间依赖系数0.5
- 测量误差: t(3)分布，时间依赖系数0.3

## 运行方法

### 1. 快速演示（Demo模式）

运行单次试验，生成可视化结果：

```bash
cd /Users/bleachvex/Downloads/projects/qtsnn
python benchmark_st.py
# 或者显式指定 --demo
python benchmark_st.py --demo
```

这将：
- 运行1次试验
- 在3个场景上测试模型
- 生成真实分位数和预测分位数的热图
- 结果保存在 `plots/` 目录

### 2. 完整实验

使用 `--full` 参数运行完整实验：

```bash
python benchmark_st.py --full
```

这将：
- 运行100次试验
- 测试不同样本量: [1000, 5000, 10000]
- 测试5个分位数: [0.05, 0.25, 0.5, 0.75, 0.95]
- 结果保存在 `results/st_mse_results.npy`

### 3. 自定义参数

可以修改 `benchmark_st.py` 中的参数：

```python
N_trials = 100              # 试验次数
N_test = 10000              # 测试集大小
sample_sizes = [1000, 5000, 10000]  # 训练集大小
quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])  # 分位数
```

## 输出结果

### 1. 可视化结果 (plots/)

- `st_scenario{X}-quantile{Y}-truth.pdf`: 真实分位数热图
- `st_scenario{X}-quantile{Y}-n{N}-{model}.pdf`: 模型预测热图

### 2. 数值结果 (results/)

- `st_mse_results.npy`: 完整的MSE结果数组
  - 维度: (N_trials, N_scenarios, N_models, N_sample_sizes, N_quantiles)

### 3. 运行日志 (logs/)

- `benchmark_st_demo_YYYY-MM-DD-HH-MM-SS.log`: Demo模式运行日志
- `benchmark_st_full_YYYY-MM-DD-HH-MM-SS.log`: 完整实验运行日志

所有打印输出都会同时显示在控制台和保存到日志文件中。

### 3. 读取结果

```python
import numpy as np

# 加载结果
results = np.load('results/st_mse_results.npy')

# 计算平均MSE
mean_mse = np.nanmean(results, axis=0)
print(mean_mse)

# 查看日志
# 日志文件在 logs/ 目录下，按时间戳命名
```

## 模型说明

项目包含两个模型：

1. **SqErrNetwork**: 平方误差神经网络（基线）
   - 优化均方误差
   - 预测条件均值

2. **QuantileNetwork**: 分位数神经网络
   - 优化分位数损失
   - 同时预测多个分位数
   - 强制分位数单调性

## 神经网络架构

- 输入层: d维特征
- 隐藏层: 200 → 200 (带Dropout和BatchNorm)
- 输出层: n_quantiles维
- 激活函数: ReLU
- 优化器: SGD with Nesterov momentum

## 时空数据特性

数据生成过程包含：

1. **时间依赖**: 
   - 协变量在时间步之间有10%的持续性
   - 误差项通过AR(1)过程相关

2. **空间依赖**:
   - 通过正弦基函数建模
   - delta函数捕获空间相关性

3. **变化的观测数量**:
   - 每个时间点的观测数量不同
   - m[t] ∈ {10, 16, 20, 24} × m_mult

## 注意事项

1. 首次运行可能需要较长时间（神经网络训练）
2. 建议先运行demo模式验证环境配置
3. 完整实验建议在GPU上运行
4. 结果会自动保存，可以中断后继续

## 引用

基于以下论文的实现：
- Quantile regression with ReLU Networks (Padilla, Tansey, Chen)
- QTS项目的时空数据生成方法

## 联系方式

如有问题，请检查：
1. 依赖包是否正确安装
2. Python版本 >= 3.7
3. PyTorch是否正确安装



# 日志功能说明

## 自动日志记录

运行 `benchmark_st.py` 时，所有输出会自动保存到日志文件：

```bash
# Demo模式
python benchmark_st.py
# 日志保存到: logs/benchmark_st_demo_2024-01-15-10-30-45.log

# 完整实验
python benchmark_st.py --full
# 日志保存到: logs/benchmark_st_full_2024-01-15-10-30-45.log
```

## 日志内容

日志文件包含：
- 实验配置信息
- 每个trial的详细输出
- 模型训练进度
- MSE结果
- 最终统计信息

## 查看日志

```bash
# 查看最新日志
ls -lt logs/ | head -5

# 实时查看日志（在另一个终端）
tail -f logs/benchmark_st_demo_*.log

# 搜索特定内容
grep "MSE" logs/benchmark_st_demo_*.log
```

## 日志文件命名

格式: `benchmark_st_{mode}_{timestamp}.log`
- `mode`: demo 或 full
- `timestamp`: YYYY-MM-DD-HH-MM-SS

## 示例输出

```
================================================================================
Quantile Regression Neural Network - Spatial-Temporal Benchmarks
================================================================================
Mode: Demo (1 trial)
Time: 2024-01-15-10-30-45
Log file: logs/benchmark_st_demo_2024-01-15-10-30-45.log
================================================================================

Results shape: (1, 3, 2, 3, 5)
Trial 1/1
  Scenario 1: STScenario1
    N=1000
      SqErr Network
      Quantile Network
...
```
