# rust-skrf 代码审查报告

> Linus Torvalds 风格审查 - 2025-12-10

---

## 总体评价

【品味评分】🟡 **凑合**

代码功能实现正确，测试通过，但存在明显的架构级技术债务。主要问题是 **矩阵库混用** 和 **巨型函数**。

---

## 致命问题

### 1. ndarray + nalgebra 双重矩阵库混用 🔴

**问题描述：**

项目同时使用两个矩阵库：
- `ndarray` - 用于数据存储和基本操作
- `nalgebra` - 用于线性代数（SVD, QR, 特征值）

这导致大量代码仅用于类型转换：

```rust
// passivity.rs 第 293-317 行 - 24 行代码只为了矩阵求逆
fn invert_complex_matrix(a: &Array2<Complex64>) -> Option<Array2<Complex64>> {
    use nalgebra::DMatrix;
    
    let (m, n) = a.dim();
    // 转换到 nalgebra
    let na_matrix = DMatrix::from_fn(m, n, |i, j| {
        nalgebra::Complex::new(a[[i, j]].re, a[[i, j]].im)
    });
    
    // 实际操作只有这一行
    match na_matrix.try_inverse() {
        Some(inv) => {
            // 转换回 ndarray (又是 N^2 次循环)
            let mut result = Array2::<Complex64>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    result[[i, j]] = Complex64::new(inv[(i, j)].re, inv[(i, j)].im);
                }
            }
            Some(result)
        }
        None => None,
    }
}
```

**影响：**
- 性能损失：每次线性代数操作都有 O(n²) 的转换开销
- 代码膨胀：~100 行代码仅用于类型转换
- 维护困难：两套 API，两套心智模型

**Linus 式评论：**
> "这就像用两种语言写一个程序，然后雇一个翻译在中间来回传话。翻译不会让你的程序更快，只会让它更慢、更难维护。"

---

### 2. 巨型函数 🔴

| 函数 | 行数 | 违反原则 |
|------|------|----------|
| `pole_relocation` | 290 行 | 函数只做一件事 |
| `passivity_enforce` | 210 行 | 函数只做一件事 |
| `fit_residues` | 110 行 | 勉强接受 |

**pole_relocation 函数分析：**

```rust
// algorithms.rs 第 106-397 行
pub fn pole_relocation(...) -> Result<PoleRelocationResult, String> {
    // 第 1 部分：构建 s = j*omega (10 行)
    // 第 2 部分：计算权重 (15 行)
    // 第 3 部分：分离实/复极点 (20 行)
    // 第 4 部分：构建系数矩阵 (80 行)      <- 应该是单独函数
    // 第 5 部分：QR 分解 (30 行)           <- 应该是单独函数
    // 第 6 部分：构建 A_fast (25 行)
    // 第 7 部分：最小二乘求解 (30 行)      <- 应该是单独函数
    // 第 8 部分：构建 H 矩阵 (40 行)       <- 应该是单独函数
    // 第 9 部分：特征值提取 (10 行)
    ...
}
```

**Linus 式评论：**
> "如果你需要滚动 5 屏才能看完一个函数，你的函数就太长了。"

---

### 3. 重复的极点类型判断 🟡

同一模式在代码中出现 **10+ 次**：

```rust
// 模式 1：计算数量
let n_poles_real = poles.iter().filter(|p| p.im == 0.0).count();
let n_poles_cmplx = poles.iter().filter(|p| p.im != 0.0).count();

// 模式 2：分支处理
for pole in poles.iter() {
    if pole.im == 0.0 {
        // 实极点处理
    } else {
        // 复极点处理
    }
}
```

**建议：** 创建一个 `PoleSet` 结构体：

```rust
struct PoleSet {
    real_poles: Vec<f64>,           // 实极点（只存实部）
    complex_poles: Vec<Complex64>,  // 复极点（只存正虚部的）
}

impl PoleSet {
    fn from_poles(poles: &[Complex64]) -> Self { ... }
    fn model_order(&self) -> usize { ... }
    fn iter_with_type(&self) -> impl Iterator<Item = (PoleType, Complex64)> { ... }
}
```

---

## 第三方库建议

### 最终决策：使用 faer 替代 nalgebra

经过评估，**faer** 是本项目的最佳选择。

### 为什么选择 faer 而不是 ndarray-linalg？

| 维度 | ndarray-linalg | faer | 胜者 |
|------|----------------|------|------|
| **WASM 支持** | ❌ 需要 BLAS (不支持) | ✅ 纯 Rust (原生支持) | faer |
| **系统依赖** | 需要 OpenBLAS/MKL | 零依赖 | faer |
| **Windows 构建** | 痛苦 (需配置 BLAS) | `cargo build` 即可 | faer |
| **性能** | 依赖 BLAS 实现 | 接近 OpenBLAS | 平手 |
| **API 现代性** | 传统风格 | 现代 Rust 风格 | faer |

**关键理由：**

1. **WASM 兼容性是硬性要求**
   - 本项目包含 `skrf-wasm` 模块
   - `ndarray-linalg` 依赖 BLAS/LAPACK (C/Fortran)
   - 为 WASM 编译 BLAS 是一场噩梦
   - `faer` 是纯 Rust，`wasm32-unknown-unknown` 开箱即用

2. **开发者体验**
   - `faer`: `cargo add faer` → 直接使用
   - `ndarray-linalg`: 安装 OpenBLAS → 配置环境变量 → 祈祷

3. **"Keep it simple"**
   > "如果你的构建系统比你的代码还复杂，你就走错路了。" — Linus 风格

### faer 迁移示例

```toml
# Cargo.toml
[dependencies]
faer = "0.23"
```

```rust
use faer::prelude::*;

// 矩阵求逆 (当前 24 行 → 3 行)
fn invert_complex_matrix(a: &Array2<Complex64>) -> Option<Array2<Complex64>> {
    let mat = Mat::from_fn(m, n, |i, j| c64::new(a[[i,j]].re, a[[i,j]].im));
    let inv = mat.partial_piv_lu().solve(&Mat::identity(m, n));
    // 转回 ndarray...
}

// SVD 分解
let svd = matrix.thin_svd();
let u = svd.u();
let s = svd.s_diagonal();
let v = svd.v();

// QR 分解
let qr = matrix.col_piv_qr();

// 特征值
let eigs = matrix.complex_eigenvalues();
```

### 迁移策略

1. **第一步**：在 `skrf-core/Cargo.toml` 添加 `faer = "0.23"`
2. **第二步**：创建 `linalg.rs` 封装层，隔离 faer API
3. **第三步**：逐个替换 `passivity.rs` 和 `algorithms.rs` 中的 nalgebra 调用
4. **第四步**：移除 `nalgebra` 依赖
5. **验证**：运行 `cargo test` + `cargo build --target wasm32-unknown-unknown`

### rayon 并行化 (后续优化)

```rust
use rayon::prelude::*;

// 当前串行实现
for f_idx in 0..n_samples {
    // 计算每个频率点的 S 矩阵
}

// 并行化实现
(0..n_samples).into_par_iter().for_each(|f_idx| {
    // 并行计算每个频率点的 S 矩阵
});
```

---

## 代码质量问题

### 1. 命名问题 🟡

```rust
// 不好的命名
let a_fast = Array2::<f64>::zeros((dim0, n_cols_used));
let prod_neg = ss.b.dot(&inv_d_minus_i).dot(&ss.c);
let h_matrix = Array2::<f64>::zeros((h_size, h_size));

// 更好的命名
let compressed_system_matrix = Array2::<f64>::zeros((dim0, n_cols_used));
let feedback_term_negative = ss.b.dot(&inv_d_minus_i).dot(&ss.c);
let pole_extraction_matrix = Array2::<f64>::zeros((h_size, h_size));
```

### 2. 魔法数字 🟡

```rust
// passivity.rs
let delta_threshold = 0.999;  // 什么是 0.999？为什么不是 0.99 或 0.9999？
let perturbation = update / count as f64 * 0.1;  // 0.1 从哪来的？

// 建议
const PASSIVITY_THRESHOLD: f64 = 0.999;  // 接近于 1 的被动性阈值
const DAMPING_FACTOR: f64 = 0.1;         // 收敛速度与稳定性的平衡
```

### 3. 错误处理不一致 🟡

```rust
// 有些地方返回 Result
fn passivity_test(...) -> Result<PassivityTestResult, String> { ... }

// 有些地方返回 Option
fn group_delay(&self) -> Option<Array3<f64>> { ... }

// 有些地方直接 continue 跳过错误
if let Some(inv) = invert_complex_matrix(&s_minus_a) {
    // ...
}  // 如果矩阵不可逆，就静默跳过这个频率点？
```

---

## 好的方面 ✅

1. **测试覆盖良好**
   - 每个模块都有对应的测试文件
   - 测试与 Python scikit-rf 结果对比

2. **文档注释完整**
   - 所有公开 API 都有 docstring
   - 包含参数说明和使用示例

3. **模块化设计**
   - 功能按职责拆分到不同文件
   - 清晰的公开 API

4. **类型安全**
   - 充分利用 Rust 类型系统
   - 编译期捕获错误

---

## 重构优先级

| 优先级 | 任务 | 工作量 | 影响 |
|--------|------|--------|------|
| P0 | 统一矩阵库（选择 ndarray-linalg 或 faer） | 3-5 天 | 性能 + 可维护性 |
| P1 | 拆分 pole_relocation 函数 | 1 天 | 可读性 |
| P1 | 创建 PoleSet 抽象 | 0.5 天 | 消除重复代码 |
| P2 | 并行化频率循环 | 1 天 | 性能 |
| P2 | 统一错误处理 | 0.5 天 | 一致性 |

---

## 结论

【核心判断】
✅ **已决策**：迁移到 faer（WASM 支持是硬性要求）

【关键洞察】
- **数据结构**：ndarray 保留用于数据存储，faer 用于线性代数
- **复杂度**：消除 nalgebra↔ndarray 转换代码（~150行）
- **风险点**：faer API 与 nalgebra 不同，需仔细对照文档

【Linus式方案】
1. 第一步：创建 `linalg.rs` 封装层
2. 第二步：逐模块迁移，每次迁移后运行全部测试
3. 第三步：移除 nalgebra 依赖
4. 确保零破坏性：保持公开 API 不变

