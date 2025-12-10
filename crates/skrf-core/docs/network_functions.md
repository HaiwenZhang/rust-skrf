# network.py 函数分类目录

---

## 1. `core.rs` - 构造器与基础字段

### Network 类方法

| 函数                         | 签名                                                      | 说明             |
| ---------------------------- | --------------------------------------------------------- | ---------------- |
| `__init__`                   | `(file, name, params, comments, f_unit, s_def, **kwargs)` | 构造器           |
| `from_z`                     | `(cls, z, *args, **kw)`                                   | 从 Z 参数创建    |
| `copy`                       | `(shallow_copy=False)`                                    | 复制网络         |
| `copy_from`                  | `(other)`                                                 | 从另一个网络复制 |
| `copy_subset`                | `(key)`                                                   | 复制频率子集     |
| `nports` / `number_of_ports` | `()`                                                      | 端口数           |
| `frequency`                  | getter/setter                                             | 频率对象         |
| `f`                          | getter/setter                                             | 频率向量 (Hz)    |
| `z0`                         | getter/setter                                             | 参考阻抗         |
| `name`                       | 属性                                                      | 网络名称         |
| `comments`                   | 属性                                                      | 注释             |
| `port_modes`                 | getter/setter                                             | 端口模式         |
| `port_tuples`                | `()`                                                      | 端口索引对列表   |

---

## 2. `params.rs` - 参数属性

### 主参数 (getter/setter)

| 参数         | 说明                     |
| ------------ | ------------------------ |
| `s`          | S 参数 (散射)            |
| `z`          | Z 参数 (阻抗)            |
| `y`          | Y 参数 (导纳)            |
| `t`          | T 参数 (散射传输)        |
| `a` / `abcd` | ABCD 参数                |
| `h`          | H 参数 (混合)            |
| `g`          | G 参数 (逆混合)          |
| `s_invert`   | 反转 S 参数              |
| `inv`        | 逆 S 矩阵 (de-embedding) |

### S 参数变体

| 属性          | 说明       |
| ------------- | ---------- |
| `s_traveling` | 行波定义   |
| `s_power`     | 功率波定义 |
| `s_pseudo`    | 伪波定义   |

---

## 3. `derived.rs` - 派生属性

### S 参数格式转换

| 函数           | 说明              |
| -------------- | ----------------- |
| `s_db`         | S 参数 (dB)       |
| `s_mag`        | S 参数幅度        |
| `s_deg`        | S 参数相位 (度)   |
| `s_rad`        | S 参数相位 (弧度) |
| `s_re`         | S 参数实部        |
| `s_im`         | S 参数虚部        |
| `s_arcl`       | 弧长              |
| `s_deg_unwrap` | 展开相位          |
| `s_time`       | 时域表示          |
| `s_time_db`    | 时域 (dB)         |
| `s_time_step`  | 阶跃响应          |

### 其他派生值

| 属性                           | 说明         |
| ------------------------------ | ------------ |
| `vswr`                         | 电压驻波比   |
| `group_delay`                  | 群时延       |
| `passivity`                    | 无源性度量   |
| `reciprocity` / `reciprocity2` | 互易性度量   |
| `stability`                    | 稳定性因子 K |
| `max_stable_gain`              | 最大稳定增益 |
| `max_gain`                     | 最大可用增益 |
| `unilateral_gain`              | 单边增益     |

---

## 4. `operators.rs` - 网络运算

### 运算符重载

| 运算符 | 方法                      | 说明         |
| ------ | ------------------------- | ------------ |
| `**`   | `__pow__`                 | cascade 级联 |
| `>>`   | `__rshift__`              | 4-port 级联  |
| `//`   | `__floordiv__`            | de-embed     |
| `*`    | `__mul__` / `__rmul__`    | 逐元素乘     |
| `+`    | `__add__` / `__radd__`    | 逐元素加     |
| `-`    | `__sub__` / `__rsub__`    | 逐元素减     |
| `/`    | `__div__` / `__truediv__` | 逐元素除     |
| `==`   | `__eq__`                  | 相等         |
| `!=`   | `__ne__`                  | 不等         |
| `[]`   | `__getitem__`             | 切片/索引    |

### 端口操作

| 函数         | 签名                     | 说明               |
| ------------ | ------------------------ | ------------------ |
| `flip`       | `()`                     | 翻转端口 (inplace) |
| `flipped`    | `()`                     | 返回翻转后的网络   |
| `renumber`   | `(from_ports, to_ports)` | 重新编号 (inplace) |
| `renumbered` | `(from_ports, to_ports)` | 返回重新编号的网络 |
| `subnetwork` | `(ports, offby=1)`       | 提取子网络         |

---

## 5. `connect.rs` - 端口连接

### Network 方法

| 函数                         | 签名 | 说明 |
| ---------------------------- | ---- | ---- |
| _(无直接方法，使用独立函数)_ |      |      |

### 独立函数

| 函数              | 签名                          | 说明         |
| ----------------- | ----------------------------- | ------------ |
| `connect`         | `(ntwkA, k, ntwkB, l, num=1)` | 连接两个网络 |
| `connect_fast`    | `(ntwkA, k, ntwkB, l)`        | connect 别名 |
| `innerconnect`    | `(ntwkA, k, l, num=1)`        | 内部端口连接 |
| `parallelconnect` | `(ntwks, ports, name=None)`   | 并联连接     |
| `cascade`         | _(通过 `**` 运算符)_          | 级联         |
| `stitch`          | `(ntwkA, ntwkB, **kwargs)`    | 频率轴拼接   |

### S 矩阵操作函数

| 函数                   | 签名                  | 说明           |
| ---------------------- | --------------------- | -------------- |
| `connect_s`            | `(A, k, B, l, num=1)` | S 矩阵连接     |
| `innerconnect_s`       | `(A, k, l)`           | S 矩阵内部连接 |
| `innerconnect_s_lstsq` | `(A, k, l)`           | 最小二乘法连接 |

---

## 6. `interpolation.rs` - 频率插值

### Network 方法

| 函数                            | 签名                                 | 说明             |
| ------------------------------- | ------------------------------------ | ---------------- |
| `interpolate`                   | `(freq_or_n, basis='s', ...)`        | 频率插值         |
| `interpolate_self`              | `(freq_or_n, **kwargs)`              | 原地插值         |
| `extrapolate_to_dc`             | `(points=None, dc_sparam=None, ...)` | 外推到 DC        |
| `crop`                          | `(f_start, f_stop, unit=None)`       | 裁剪 (inplace)   |
| `cropped`                       | `(f_start, f_stop, unit=None)`       | 返回裁剪后的网络 |
| `resample`                      | `(npoints, **kwargs)`                | 重采样           |
| `drop_non_monotonic_increasing` | `()`                                 | 删除非单调递增点 |

### 独立函数

| 函数            | 签名             | 说明         |
| --------------- | ---------------- | ------------ |
| `overlap`       | `(ntwkA, ntwkB)` | 频率重叠部分 |
| `overlap_multi` | `(ntwk_list)`    | 多网络重叠   |

---

## 7. `properties.rs` - 属性检验

### Network 方法

| 函数             | 签名                              | 说明         |
| ---------------- | --------------------------------- | ------------ |
| `is_reciprocal`  | `(tol=ALMOST_ZERO)`               | 互易性检验   |
| `is_symmetric`   | `(n=1, port_order=None, tol=...)` | 对称性检验   |
| `is_passive`     | `(tol=ALMOST_ZERO)`               | 无源性检验   |
| `is_lossless`    | `(tol=ALMOST_ZERO)`               | 无损耗检验   |
| `nonreciprocity` | `(m, n, normalize=False)`         | 非互易性度量 |
| `s_error`        | `(ntwk, error_function)`          | S 参数误差   |

### 独立函数

| 函数          | 签名                             | 说明                |
| ------------- | -------------------------------- | ------------------- |
| `passivity`   | `(s)`                            | 无源性度量 (S 矩阵) |
| `reciprocity` | `(s)`                            | 互易性度量 (S 矩阵) |
| `s_error`     | `(ntwkA, ntwkB, error_function)` | 误差函数            |

### 检验断言

| 函数                       | 签名                   |
| -------------------------- | ---------------------- |
| `check_frequency_equal`    | `(ntwkA, ntwkB)`       |
| `check_frequency_exist`    | `(ntwk)`               |
| `check_z0_equal`           | `(ntwkA, ntwkB)`       |
| `check_nports_equal`       | `(ntwkA, ntwkB)`       |
| `assert_frequency_equal`   | `(ntwkA, ntwkB)`       |
| `assert_frequency_exist`   | `(ntwk)`               |
| `assert_z0_equal`          | `(ntwkA, ntwkB)`       |
| `assert_z0_at_ports_equal` | `(ntwkA, k, ntwkB, l)` |
| `assert_nports_equal`      | `(ntwkA, ntwkB)`       |

---

## 8. `noise.rs` - 噪声参数 (可选)

### Network 方法

| 函数          | 签名                                    | 说明               |
| ------------- | --------------------------------------- | ------------------ |
| `noisy`       | `()`                                    | 是否有噪声数据     |
| `n`           | getter                                  | 噪声相关矩阵       |
| `f_noise`     | getter                                  | 噪声频率向量       |
| `y_opt`       | getter                                  | 最优源导纳         |
| `z_opt`       | getter                                  | 最优源阻抗         |
| `g_opt`       | getter                                  | 最优源反射系数     |
| `nfmin`       | getter                                  | 最小噪声系数       |
| `nfmin_db`    | getter                                  | 最小噪声系数 (dB)  |
| `nf`          | `(z)`                                   | 指定阻抗的噪声系数 |
| `nfdb_gs`     | `(gs)`                                  | 噪声系数 vs 源     |
| `rn`          | getter                                  | 等效噪声电阻       |
| `set_noise_a` | `(noise_freq, nfmin_db, gamma_opt, rn)` | 设置噪声           |
| `nf_circle`   | `(nf, npoints=181)`                     | 噪声系数圆         |

---

## 9. `time_domain.rs` - 时域分析 (可选)

### Network 方法

| 函数               | 签名                                     | 说明     |
| ------------------ | ---------------------------------------- | -------- |
| `impulse_response` | `(window='hamming', n=None, pad=0, ...)` | 冲激响应 |
| `step_response`    | `(window='hamming', n=None, pad=0, ...)` | 阶跃响应 |

---

## 10. `mixed_mode.rs` - 混合模式转换 (可选)

### Network 方法

| 函数        | 签名                          | 说明            |
| ----------- | ----------------------------- | --------------- |
| `se2gmm`    | `(p, z0_mm=None, s_def=None)` | 单端 → 混合模式 |
| `gmm2se`    | `(p, z0_se=None, s_def=None)` | 混合模式 → 单端 |
| `_m`        | 内部                          |                 |
| `_M`        | 内部                          |                 |
| `_M_circle` | 内部                          |                 |
| `_X`        | 内部                          |                 |
| `_P`        | 内部                          |                 |
| `_Q`        | 内部                          |                 |
| `_Xi`       | 内部                          |                 |
| `_Xi_tilde` | 内部                          |                 |

### 独立函数

| 函数            | 签名                                            | 说明           |
| --------------- | ----------------------------------------------- | -------------- |
| `evenodd2delta` | `(n, z0=50, renormalize=True, doublehalf=True)` | 偶/奇模 → 正常 |

---

## 11. `active.rs` - 有源参数 (可选)

### Network 方法

| 函数               | 签名                               | 说明        |
| ------------------ | ---------------------------------- | ----------- |
| `s_active`         | `(a)`                              | 有源 S 参数 |
| `z_active`         | `(a)`                              | 有源 Z 参数 |
| `y_active`         | `(a)`                              | 有源 Y 参数 |
| `vswr_active`      | `(a)`                              | 有源 VSWR   |
| `stability_circle` | `(target_port, npoints=181)`       | 稳定性圆    |
| `gain_circle`      | `(target_port, gain, npoints=181)` | 增益圆      |

### 独立函数

| 函数            | 签名         |
| --------------- | ------------ |
| `s2s_active`    | `(s, a)`     |
| `s2z_active`    | `(s, z0, a)` |
| `s2y_active`    | `(s, z0, a)` |
| `s2vswr_active` | `(s, a)`     |

---

## 12. `math/transforms.rs` - 参数转换 (已存在)

### 独立函数

| 函数            | 签名                                  | 说明       |
| --------------- | ------------------------------------- | ---------- |
| `s2z`           | `(s, z0=50, s_def=...)`               | S → Z      |
| `s2y`           | `(s, z0=50, s_def=...)`               | S → Y      |
| `s2t`           | `(s)`                                 | S → T      |
| `s2s`           | `(s, z0, s_def_new, s_def_old)`       | S 定义转换 |
| `z2s`           | `(z, z0=50, s_def=...)`               | Z → S      |
| `z2y`           | `(z)`                                 | Z → Y      |
| `z2t`           | `(z)`                                 | Z → T      |
| `y2s`           | `(y, z0=50, s_def=...)`               | Y → S      |
| `y2z`           | `(y)`                                 | Y → Z      |
| `y2t`           | `(y)`                                 | Y → T      |
| `t2s`           | `(t)`                                 | T → S      |
| `t2z`           | `(t, z0=50)`                          | T → Z      |
| `t2y`           | `(t, z0=50)`                          | T → Y      |
| `h2s`           | `(h, z0=50)`                          | H → S      |
| `s2h`           | `(s, z0=50)`                          | S → H      |
| `z2h`           | `(z)`                                 | Z → H      |
| `g2s`           | `(g, z0=50)`                          | G → S      |
| `s2g`           | `(s, z0=50)`                          | S → G      |
| `renormalize_s` | `(s, z_old, z_new, s_def, s_def_old)` | 重新归一化 |

---

## 13. I/O 函数 (不移植到 Rust 核心)

### Network 方法

| 函数                   | 说明             |
| ---------------------- | ---------------- |
| `read_touchstone`      | 读取 Touchstone  |
| `zipped_touchstone`    | 从 ZIP 读取      |
| `write_touchstone`     | 写入 Touchstone  |
| `write`                | 写入 pickle      |
| `read`                 | 读取 pickle      |
| `write_spreadsheet`    | 写入 Excel       |
| `to_dataframe`         | 转换为 DataFrame |
| `write_to_json_string` | 序列化为 JSON    |
| `_write_noisedata`     | 内部噪声数据写入 |

---

## 14. Plotting 函数 (不移植)

### Network 方法

| 函数                           | 说明       |
| ------------------------------ | ---------- |
| `plot`                         | 通用绑定   |
| `plot_attribute`               | 属性图     |
| `plot_s_db`, `plot_s_deg`, ... | S 参数图   |
| `plot_s_smith`                 | Smith 圆图 |
| `plot_passivity`               | 无源性图   |
| `plot_reciprocity`             | 互易性图   |
| `plot_it_all`                  | 全图       |

---

## 15. 网络组合函数

### 独立函数

| 函数                      | 签名                               | 说明                 |
| ------------------------- | ---------------------------------- | -------------------- |
| `average`                 | `(list_of_networks, polar=False)`  | 平均                 |
| `stdev`                   | `(list_of_networks, attr='s')`     | 标准差               |
| `concat_ports`            | `(ntwk_list, port_order='second')` | 端口拼接             |
| `one_port_2_two_port`     | `(ntwk)`                           | 1-port → 2-port      |
| `four_oneports_2_twoport` | `(s11, s12, s21, s22)`             | 4×1-port → 2-port    |
| `n_oneports_2_nport`      | `(ntwk_list)`                      | N×1-port → N-port    |
| `n_twoports_2_nport`      | `(ntwk_list, nports, offby=1)`     | 多个 2-port → N-port |
| `twoport_to_nport`        | `(ntwk, port1, port2, nports)`     | 2-port → N-port      |
| `chopinhalf`              | `(ntwk)`                           | 切半                 |
| `subnetwork`              | `(ntwk, ports, offby=1)`           | 子网络               |

---

## 16. 工具函数

| 函数                 | 签名                   | 说明            |
| -------------------- | ---------------------- | --------------- |
| `inv`                | `(s)`                  | S 矩阵逆        |
| `flip`               | `(a)`                  | S 矩阵翻转      |
| `fix_param_shape`    | `(p)`                  | 修正参数形状    |
| `fix_z0_shape`       | `(z0, nfreqs, nports)` | 修正 Z0 形状    |
| `impedance_mismatch` | `(z1, z2, s_def)`      | 阻抗失配 S 矩阵 |
| `two_port_reflect`   | `(ntwk1, ntwk2, name)` | 反射两端网络    |

---

## 统计

| 类别      | 方法/函数数 | 目标模块                |
| --------- | ----------- | ----------------------- |
| 构造/基础 | 13          | `core.rs`               |
| 参数属性  | 12          | `params.rs`             |
| 派生属性  | 18          | `derived.rs`            |
| 运算符    | 15          | `operators.rs`          |
| 端口连接  | 8           | `connect.rs`            |
| 频率插值  | 9           | `interpolation.rs`      |
| 属性检验  | 15          | `properties.rs`         |
| 噪声      | 13          | `noise.rs` (可选)       |
| 时域      | 2           | `time_domain.rs` (可选) |
| 混合模式  | 11          | `mixed_mode.rs` (可选)  |
| 有源参数  | 10          | `active.rs` (可选)      |
| 参数转换  | 19          | `math/transforms.rs`    |
| I/O       | 9           | touchstone/ + 不移植    |
| Plotting  | ~15         | 不移植                  |
| 网络组合  | 10          | 独立模块                |
| 工具      | 6           | 分散各模块              |
| **总计**  | **~185**    |                         |
