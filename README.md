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

nohup python benchmark_st_wrapped.py --scenarios 6 7 8 --demo &  
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




训练数据 (y_train)
y_train = func.sample(X_train)
# 例如: noiseless(X) + np.random.standard_t(3, size=X.shape[0])

Copy
每个样本的误差是随机的，有正有负

例如：[10.5, 9.8, 11.2, 8.9, 10.3, ...] （假设 noiseless = 10）

模型从这些带随机噪声的数据中学习

测试/评估 (y_quantiles)
y_quantiles = np.array([func.quantile(X_test, q) for q in quantiles]).T
# 例如对于 q=0.05, 0.5, 0.95:
# quantile(X, 0.05) = noiseless(X) + t_dist.ppf(0.05, 3)  # 固定负值
# quantile(X, 0.50) = noiseless(X) + t_dist.ppf(0.50, 3)  # = 0
# quantile(X, 0.95) = noiseless(X) + t_dist.ppf(0.95, 3)  # 固定正值

Copy
python
对于每个 q，误差是确定的

例如（假设 noiseless = 10）：

q=0.05: 10 + (-2.35) = 7.65 （5% 分位数）

q=0.50: 10 + 0 = 10.0 （中位数）

q=0.95: 10 + 2.35 = 12.35 （95% 分位数）

具体例子
假设 t(df=3) 分布：

# 训练时 (sample)：随机噪声
np.random.standard_t(3, size=5)
# 输出: [0.8, -1.2, 2.1, -0.5, 0.3]  # 每次都不同！

# 测试时 (quantile)：固定分位数
t_dist.ppf(0.05, 3)  # 总是 -2.353
t_dist.ppf(0.50, 3)  # 总是 0.0
t_dist.ppf(0.95, 3)  # 总是 2.353

Copy
python
为什么这样设计？
训练：模型需要从真实的随机数据中学习噪声的分布

评估：我们想知道模型能否预测理论上的分位数（ground truth）

这就像：

训练：给学生看很多考试成绩（有高有低，随机的）

测试：问学生"第5百分位是多少？中位数是多少？第95百分位是多少？"（固定答案）

所以你的理解完全正确：

sample 用随机误差生成训练数据

quantile 用固定分位数作为评估标准

那这个区别就是sqerrnetwork 和 quantilenetowrik 在模拟的时候的区别，这两个有啥区别呢？他们在哪里定义的？


Context
让我查看这两个模型的定义：

2 files read
neural_sqerr.py
neural_model.py
完美！现在我可以解释这两个模型的核心区别：

SqErrNetwork (均方误差网络)
定义位置: /Users/bleachvex/Downloads/projects/qtsnn/neural_sqerr.py

目标: 预测条件期望 E[Y|X]（即条件均值）

损失函数:

def sqerr_loss(yhat, tidx):
    return (tY[tidx,None] - yhat)**2  # 均方误差

Copy
python
输出:

单个值（期望值）

对于多个分位数，会重复输出相同的期望值

特点:

优化目标：最小化 (y - ŷ)²

预测的是平均值（中心趋势）

不考虑分位数信息

QuantileNetwork (分位数网络)
定义位置: /Users/bleachvex/Downloads/projects/qtsnn/neural_model.py

目标: 预测条件分位数 Q_τ[Y|X]（对于多个 τ 值）

损失函数:

def quantile_loss(yhat, tidx):
    z = tY[tidx,None] - yhat
    return torch.max(tquantiles[None]*z, (tquantiles[None] - 1)*z)

Copy
python
这是分位数损失（也叫 pinball loss）：

当 y > ŷ 时：损失 = τ × (y - ŷ)

当 y < ŷ 时：损失 = (1-τ) × (ŷ - y)

输出:

多个值（每个分位数一个）

例如：[q₀.₀₅, q₀.₂₅, q₀.₅, q₀.₇₅, q₀.₉₅]

特点:

优化目标：最小化分位数损失

预测整个条件分布的不同分位数

强制单调性：通过 softplus 确保 q₀.₀₅ ≤ q₀.₂₅ ≤ q₀.₅ ≤ ...

关键区别总结
特性	SqErrNetwork	QuantileNetwork
预测目标	条件期望 E[Y|X]	条件分位数 Q_τ[Y|X]
损失函数	MSE: (y-ŷ)²	Quantile loss: max(τ·z, (τ-1)·z)
输出数量	1个值	多个值（每个τ一个）
对异常值	敏感（平方惩罚）	鲁棒（线性惩罚）
信息量	只有均值	整个分布的形状
在 benchmark 中的作用
models = [lambda: SqErrNetwork(),              # 基线：只预测均值
          lambda: QuantileNetwork(quantiles)]  # 目标：预测所有分位数

Copy
python
SqErrNetwork: 作为基线，看看只预测均值的表现

QuantileNetwork: 真正的分位数回归模型，应该表现更好