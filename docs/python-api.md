# skrf-python API 文档

> 高性能 RF/微波网络分析库的 Python 绑定

## 安装

```bash
# 使用 maturin 构建
cd crates/skrf-python
maturin develop --release
```

## 模块导入

```python
import skrf_python as skrf
```

---

## Frequency 类

频率对象，表示一个频率范围。

### 构造函数

```python
skrf.Frequency(start, stop, npoints, unit="Hz", sweep_type="linear")
```

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `start` | `float` | - | 起始频率 |
| `stop` | `float` | - | 终止频率 |
| `npoints` | `int` | - | 频率点数 |
| `unit` | `str` | `"Hz"` | 频率单位: `"Hz"`, `"kHz"`, `"MHz"`, `"GHz"`, `"THz"` |
| `sweep_type` | `str` | `"linear"` | 扫频类型: `"linear"` 或 `"log"` |

### 属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `f` | `np.ndarray[float64]` | 频率数组 (Hz) |
| `f_scaled` | `np.ndarray[float64]` | 缩放后的频率数组 (按单位) |
| `start` | `float` | 起始频率 (Hz) |
| `stop` | `float` | 终止频率 (Hz) |
| `npoints` | `int` | 频率点数 |
| `center` | `float` | 中心频率 (Hz) |
| `span` | `float` | 频率跨度 (Hz) |
| `unit` | `str` | 频率单位 |
| `sweep_type` | `str` | 扫频类型 |

### 示例

```python
import skrf_python as skrf
import numpy as np

# 创建 1-10 GHz 频率范围，101 个点
freq = skrf.Frequency(1.0, 10.0, 101, unit="GHz", sweep_type="linear")

print(f"频率范围: {freq.start/1e9:.1f} - {freq.stop/1e9:.1f} GHz")
print(f"频率点数: {freq.npoints}")
print(f"中心频率: {freq.center/1e9:.1f} GHz")
print(f"频率跨度: {freq.span/1e9:.1f} GHz")

# 获取频率数组
f_hz = freq.f           # Hz
f_ghz = freq.f_scaled   # 按单位缩放 (GHz)
```

---

## Network 类

N 端口 RF 网络，包含 S 参数和其他网络参数。

### 静态方法

#### `from_touchstone(path)`

从 Touchstone 文件加载网络。

```python
Network.from_touchstone(path: str) -> Network
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `path` | `str` | Touchstone 文件路径 (`.s1p`, `.s2p`, `.s4p` 等) |

### 属性

#### 基本信息

| 属性 | 类型 | 描述 |
|------|------|------|
| `nports` | `int` | 端口数 |
| `nfreq` | `int` | 频率点数 |
| `name` | `Optional[str]` | 网络名称 |
| `frequency` | `Frequency` | 频率对象 |

#### S 参数

| 属性 | 类型 | 形状 | 描述 |
|------|------|------|------|
| `s` | `np.ndarray[complex128]` | `[nfreq, nports, nports]` | 复数 S 参数 |
| `s_db` | `np.ndarray[float64]` | `[nfreq, nports, nports]` | S 参数幅度 (dB) |
| `s_mag` | `np.ndarray[float64]` | `[nfreq, nports, nports]` | S 参数幅度 (线性) |
| `s_deg` | `np.ndarray[float64]` | `[nfreq, nports, nports]` | S 参数相位 (度) |
| `s_rad` | `np.ndarray[float64]` | `[nfreq, nports, nports]` | S 参数相位 (弧度) |
| `s_re` | `np.ndarray[float64]` | `[nfreq, nports, nports]` | S 参数实部 |
| `s_im` | `np.ndarray[float64]` | `[nfreq, nports, nports]` | S 参数虚部 |

#### 其他参数

| 属性 | 类型 | 形状 | 描述 |
|------|------|------|------|
| `z` | `np.ndarray[complex128]` | `[nfreq, nports, nports]` | Z 参数 (阻抗) |
| `y` | `np.ndarray[complex128]` | `[nfreq, nports, nports]` | Y 参数 (导纳) |
| `z0` | `np.ndarray[complex128]` | `[nports]` | 参考阻抗 |
| `f` | `np.ndarray[float64]` | `[nfreq]` | 频率数组 (Hz) |
| `vswr` | `np.ndarray[float64]` | `[nfreq, nports, nports]` | 电压驻波比 |

### 方法

#### `is_reciprocal(tol=None)`

检查网络是否为互易网络。

```python
is_reciprocal(tol: Optional[float] = None) -> bool
```

#### `is_passive(tol=None)`

检查网络是否为无源网络。

```python
is_passive(tol: Optional[float] = None) -> bool
```

#### `is_lossless(tol=None)`

检查网络是否为无损网络。

```python
is_lossless(tol: Optional[float] = None) -> bool
```

#### `is_symmetric(tol=None)`

检查网络是否为对称网络。

```python
is_symmetric(tol: Optional[float] = None) -> bool
```

### 示例

```python
import skrf_python as skrf
import matplotlib.pyplot as plt

# 从 Touchstone 文件加载网络
nw = skrf.Network.from_touchstone("filter.s2p")

print(f"端口数: {nw.nports}")
print(f"频率点数: {nw.nfreq}")
print(f"频率范围: {nw.f[0]/1e9:.3f} - {nw.f[-1]/1e9:.3f} GHz")

# 检查网络特性
print(f"互易性: {nw.is_reciprocal()}")
print(f"无源性: {nw.is_passive()}")
print(f"无损性: {nw.is_lossless()}")

# 获取 S 参数
s11_db = nw.s_db[:, 0, 0]  # S11 in dB
s21_db = nw.s_db[:, 1, 0]  # S21 in dB

# 绘制 S 参数
fig, ax = plt.subplots()
ax.plot(nw.f/1e9, s11_db, label='S11')
ax.plot(nw.f/1e9, s21_db, label='S21')
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('Magnitude (dB)')
ax.legend()
ax.grid(True)
plt.show()
```

---

## VectorFitting 类

矢量拟合算法，用于 S 参数的有理函数逼近。

### 构造函数

```python
skrf.VectorFitting()
```

创建一个新的 VectorFitting 实例。

### 属性

| 属性 | 类型 | 可写 | 描述 |
|------|------|------|------|
| `poles` | `np.ndarray[complex128]` | 否 | 拟合后的极点 |
| `residues` | `np.ndarray[complex128]` | 否 | 拟合后的留数 `[n_responses, n_poles]` |
| `constant_coeff` | `np.ndarray[float64]` | 否 | 常数项系数 |
| `proportional_coeff` | `np.ndarray[float64]` | 否 | 比例项系数 |
| `max_iterations` | `int` | 是 | 最大迭代次数 |
| `max_tol` | `float` | 是 | 收敛容差 |
| `wall_clock_time` | `float` | 否 | 上次拟合耗时 (秒) |

### 方法

#### `vector_fit(network, ...)`

对网络进行矢量拟合。

```python
vector_fit(
    network: Network,
    n_poles_real: int = 2,
    n_poles_cmplx: int = 2,
    init_pole_spacing: str = "linear",
    fit_constant: bool = True,
    fit_proportional: bool = False
) -> None
```

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `network` | `Network` | - | 要拟合的网络 |
| `n_poles_real` | `int` | `2` | 初始实极点数 |
| `n_poles_cmplx` | `int` | `2` | 初始复极点对数 |
| `init_pole_spacing` | `str` | `"linear"` | 极点初始分布: `"linear"` 或 `"log"` |
| `fit_constant` | `bool` | `True` | 是否拟合常数项 |
| `fit_proportional` | `bool` | `False` | 是否拟合比例项 |

#### `get_rms_error(network, i, j)`

获取拟合模型与原始网络的 RMS 误差。

```python
get_rms_error(network: Network, i: int, j: int) -> float
```

#### `get_model_order()`

获取模型阶数 (N_real + 2 * N_complex)。

```python
get_model_order() -> int
```

#### `get_model_response(i, j, freqs)`

在指定频率点获取模型响应。

```python
get_model_response(i: int, j: int, freqs: List[float]) -> np.ndarray[complex128]
```

#### `write_spice_subcircuit_s(file, network, ...)`

将拟合结果写入 SPICE 子电路文件。

```python
write_spice_subcircuit_s(
    file: str,
    network: Network,
    fitted_model_name: str = "s_equivalent",
    create_reference_pins: bool = False
) -> None
```

#### `generate_spice_subcircuit_s(network, ...)`

生成 SPICE 子电路网表字符串。

```python
generate_spice_subcircuit_s(
    network: Network,
    fitted_model_name: str = "s_equivalent",
    create_reference_pins: bool = False
) -> str
```

#### `passivity_test(network)`

执行被动性测试，返回违反被动性的频段列表。

```python
passivity_test(network: Network) -> List[[float, float]]
```

#### `is_passive(network)`

检查拟合模型是否为被动的。

```python
is_passive(network: Network) -> bool
```

#### `passivity_enforce(network, n_samples=200)`

强制模型满足被动性条件。

```python
passivity_enforce(
    network: Network,
    n_samples: int = 200
) -> Tuple[bool, int, List[float]]
```

返回: `(success, iterations, history_max_sigma)`

### 完整示例

```python
import skrf_python as skrf
import matplotlib.pyplot as plt
import numpy as np

# 加载网络
nw = skrf.Network.from_touchstone("filter.s2p")

# 创建 VectorFitting 实例
vf = skrf.VectorFitting()

# 配置参数
vf.max_iterations = 100
vf.max_tol = 1e-6

# 执行矢量拟合
vf.vector_fit(
    nw,
    n_poles_real=3,
    n_poles_cmplx=5,
    init_pole_spacing="log",
    fit_constant=True,
    fit_proportional=False
)

# 查看结果
print(f"模型阶数: {vf.get_model_order()}")
print(f"拟合耗时: {vf.wall_clock_time:.3f}s")

# 计算 RMS 误差
for i in range(nw.nports):
    for j in range(nw.nports):
        rms = vf.get_rms_error(nw, i, j)
        print(f"S{i+1}{j+1} RMS Error: {rms:.2e}")

# 获取模型响应
freqs_new = np.linspace(nw.f[0], nw.f[-1], 1001)
s21_model = vf.get_model_response(1, 0, freqs_new.tolist())

# 绘制对比图
fig, ax = plt.subplots()
ax.plot(nw.f/1e9, nw.s_db[:, 1, 0], 'b-', label='Original S21', linewidth=2)
ax.plot(freqs_new/1e9, 20*np.log10(np.abs(s21_model)), 'r--', label='Fitted S21')
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('Magnitude (dB)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# 被动性检查
is_passive = vf.is_passive(nw)
print(f"模型是否被动: {is_passive}")

if not is_passive:
    # 执行被动性强制
    success, iterations, history = vf.passivity_enforce(nw, n_samples=200)
    print(f"被动性强制: {'成功' if success else '失败'}, 迭代 {iterations} 次")

# 生成 SPICE 网表
spice_netlist = vf.generate_spice_subcircuit_s(nw, "MyFilter")
print("SPICE Netlist Preview:")
print(spice_netlist[:500])

# 或直接保存到文件
vf.write_spice_subcircuit_s("filter_model.sp", nw, "MyFilter")
```

---

## 错误处理

所有方法在遇到错误时会抛出对应的 Python 异常：

- `ValueError` - 参数无效
- `IOError` - 文件操作失败
- `RuntimeError` - 运行时错误（如模型未拟合）

```python
try:
    nw = skrf.Network.from_touchstone("nonexistent.s2p")
except IOError as e:
    print(f"文件加载失败: {e}")

try:
    order = vf.get_model_order()  # 未拟合时调用
except RuntimeError as e:
    print(f"模型未拟合: {e}")
```
