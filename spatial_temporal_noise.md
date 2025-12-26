# Scenario对比说明

## 原始 Scenario1 (funcs.py) vs ST Scenario1 (funcs_st.py)

### 原始 Scenario1 (funcs.py)
```python
class Scenario1(Benchmark):
    def noiseless(self, X):
        return self.g2(self.g1(X))
    
    def quantile(self, X, q):
        return self.noiseless(X) + self.g3(X) * t_dist.ppf(q, 2)
    
    def sample(self, X):
        return self.noiseless(X) + self.g3(X) * np.random.standard_t(2, size=X.shape[0])
    
    def g1(self, X):
        return np.array([np.sqrt(X[:,0]) + X[:,0]*X[:,1], 
                        np.cos(2*np.pi*X[:,1])]).T
    
    def g2(self, X):
        return np.sqrt(X[:,0] + X[:,1]**2) + X[:,0]**2 * X[:,1]
    
    def g3(self, X):
        return np.linalg.norm(X - 0.5, axis=1)
```

**特点：**
- 复杂的非线性函数（g1, g2, g3组合）
- 异方差噪声：g3(X) * t(2)
- 噪声大小依赖于X的位置
- **无时空相关性**

---

### ST Scenario1 (funcs_st.py)
```python
class STScenario1(SpatialTemporalWrapper):
    def __init__(self):
        super().__init__(Scenario1(), n_time=50, temporal_corr=0.3, spatial_corr=0.5)
```

**SpatialTemporalWrapper做了什么：**
```python
def sample(self, X):
    # 1. 调用原始Scenario1.sample(X)
    y_base = self.base.sample(X)
    
    # 2. 添加时空相关噪声
    for t in range(50):  # 50个时间步
        # 空间相关：Cholesky分解
        spatial_noise = L @ z
        
        # 时间相关：AR(1)过程
        temporal_noise = 0.3 * past_error + sqrt(1-0.3^2) * randn()
        
        # 组合
        errors = 0.5 * spatial_noise + 0.5 * temporal_noise
    
    return y_base + errors
```

**新增特点：**
- ✅ 保留原始复杂函数
- ✅ 添加空间相关：nearby points有相关噪声
- ✅ 添加时间相关：AR(1)过程，系数0.3
- ✅ quantile函数不变（时空效应在噪声中）

---

## 关键区别总结

| 特性 | 原始Scenario1 | ST Scenario1 |
|------|--------------|--------------|
| 函数形式 | g2(g1(X)) | 相同 |
| 噪声分布 | g3(X) * t(2) | 相同 + 时空噪声 |
| 空间相关 | ❌ 无 | ✅ 高斯核 |
| 时间相关 | ❌ 无 | ✅ AR(1) |
| quantile | noiseless + g3*t.ppf | 相同 |

---

## 为什么这样设计？

1. **保持原始函数**：
   - 原始5个scenarios已验证工作良好
   - 不改变noiseless和quantile函数

2. **只在噪声中添加时空依赖**：
   - 空间相关：通过Cholesky分解
   - 时间相关：通过AR(1)过程
   - 这样quantile函数仍然正确

3. **与R场景的区别**：
   - R场景(6-8)：从R代码转换，复杂的时间序列生成
   - Wrapped场景(1-5)：简单包装，保持原始函数

---

## 实际效果

**原始Scenario1**：
```
y = g2(g1(X)) + g3(X) * t(2)
```

**ST Scenario1**：
```
y = g2(g1(X)) + g3(X) * t(2) + spatial_noise + temporal_noise
```

时空噪声使得：
- 相邻点的y值相关（空间）
- 连续时间步的y值相关（时间）
- 但真实分位数函数不变
