# skrf-wasm API 文档

> 高性能 RF/微波网络分析库的 WebAssembly 绑定

## 安装和构建

```bash
# 安装 wasm-pack
cargo install wasm-pack

# 构建 WASM 包
cd crates/skrf-wasm
wasm-pack build --target web

# 输出目录: pkg/
```

## 在项目中使用

### npm 包引入

```bash
npm install ../crates/skrf-wasm/pkg
```

### TypeScript 配置

```json
// tsconfig.json
{
  "compilerOptions": {
    "moduleResolution": "bundler",
    "target": "ES2020"
  }
}
```

---

## 类型定义

```typescript
// types.d.ts - 完整的 TypeScript 类型定义

declare module 'skrf-wasm' {
  /** 获取库版本 */
  export function version(): string;

  /** 频率对象 */
  export class WasmFrequency {
    constructor(
      start: number,
      stop: number,
      npoints: number,
      unit?: string,
      sweepType?: string
    );
    
    readonly f: Float64Array;
    readonly f_scaled: Float64Array;
    readonly start: number;
    readonly stop: number;
    readonly npoints: number;
    readonly center: number;
    readonly span: number;
    readonly unit: string;
  }

  /** N端口RF网络 */
  export class WasmNetwork {
    static fromTouchstoneContent(content: string, filename: string): WasmNetwork;
    
    readonly nports: number;
    readonly nfreq: number;
    readonly name: string | undefined;
    readonly f: Float64Array;
    
    getSDb(): Float64Array;
    getSMag(): Float64Array;
    getSDeg(): Float64Array;
    getSRe(): Float64Array;
    getSIm(): Float64Array;
    getSDbAt(i: number, j: number): Float64Array;
    
    isReciprocal(tol?: number): boolean;
    isPassive(tol?: number): boolean;
    isLossless(tol?: number): boolean;
    isSymmetric(tol?: number): boolean;
  }

  /** 矢量拟合 */
  export class WasmVectorFitting {
    constructor();
    
    maxIterations: number;
    maxTol: number;
    readonly wallClockTime: number;
    
    vectorFit(
      network: WasmNetwork,
      nPolesReal?: number,
      nPolesCmplx?: number,
      initPoleSpacing?: string,
      fitConstant?: boolean,
      fitProportional?: boolean
    ): void;
    
    getRmsError(network: WasmNetwork, i: number, j: number): number;
    getModelOrder(): number;
    getModelResponse(
      i: number,
      j: number,
      freqs: Float64Array
    ): { real: Float64Array; imag: Float64Array };
    
    isPassive(network: WasmNetwork): boolean;
    passivityTest(network: WasmNetwork): [number, number][];
    passivityEnforce(
      network: WasmNetwork,
      nSamples?: number
    ): { success: boolean; iterations: number; historyMaxSigma: Float64Array };
    
    generateSpiceSubcircuit(
      network: WasmNetwork,
      modelName?: string,
      createReferencePins?: boolean
    ): string;
  }
}
```

---

## WasmFrequency 类

频率对象，表示一个频率范围。

### 构造函数

```typescript
new WasmFrequency(
  start: number,         // 起始频率
  stop: number,          // 终止频率
  npoints: number,       // 频率点数
  unit?: string,         // 单位: 'Hz'|'kHz'|'MHz'|'GHz'|'THz', 默认 'Hz'
  sweepType?: string     // 扫频类型: 'linear'|'log', 默认 'linear'
)
```

### 属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `f` | `Float64Array` | 频率数组 (Hz) |
| `f_scaled` | `Float64Array` | 缩放后的频率数组 |
| `start` | `number` | 起始频率 (Hz) |
| `stop` | `number` | 终止频率 (Hz) |
| `npoints` | `number` | 频率点数 |
| `center` | `number` | 中心频率 (Hz) |
| `span` | `number` | 频率跨度 (Hz) |
| `unit` | `string` | 频率单位 |

### 示例

```typescript
import init, { WasmFrequency } from 'skrf-wasm';

async function frequencyExample() {
  await init();
  
  // 创建 1-10 GHz 频率范围
  const freq = new WasmFrequency(1, 10, 101, 'GHz', 'linear');
  
  console.log(`频率范围: ${freq.start / 1e9} - ${freq.stop / 1e9} GHz`);
  console.log(`频率点数: ${freq.npoints}`);
  console.log(`中心频率: ${freq.center / 1e9} GHz`);
  
  // 获取频率数组
  const f = freq.f;  // Float64Array in Hz
  console.log(`第一个频点: ${f[0] / 1e9} GHz`);
}
```

---

## WasmNetwork 类

N 端口 RF 网络，包含 S 参数。

### 静态方法

#### `fromTouchstoneContent(content, filename)`

从 Touchstone 文件内容创建网络（适用于浏览器环境）。

```typescript
static fromTouchstoneContent(content: string, filename: string): WasmNetwork
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `content` | `string` | Touchstone 文件内容 |
| `filename` | `string` | 文件名（用于确定端口数，如 `test.s2p`）|

### 属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `nports` | `number` | 端口数 |
| `nfreq` | `number` | 频率点数 |
| `name` | `string \| undefined` | 网络名称 |
| `f` | `Float64Array` | 频率数组 (Hz) |

### 方法

#### S 参数获取

所有 S 参数方法返回扁平化的 `Float64Array`，形状为 `[nfreq × nports × nports]`（行优先）。

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| `getSDb()` | `Float64Array` | S 参数幅度 (dB) |
| `getSMag()` | `Float64Array` | S 参数幅度 (线性) |
| `getSDeg()` | `Float64Array` | S 参数相位 (度) |
| `getSRe()` | `Float64Array` | S 参数实部 |
| `getSIm()` | `Float64Array` | S 参数虚部 |
| `getSDbAt(i, j)` | `Float64Array` | 指定 S 参数 Sij 的 dB 值 |

#### 网络特性检查

| 方法 | 参数 | 返回类型 | 描述 |
|------|------|----------|------|
| `isReciprocal(tol?)` | `tol?: number` | `boolean` | 互易性检查 |
| `isPassive(tol?)` | `tol?: number` | `boolean` | 无源性检查 |
| `isLossless(tol?)` | `tol?: number` | `boolean` | 无损性检查 |
| `isSymmetric(tol?)` | `tol?: number` | `boolean` | 对称性检查 |

### 完整示例

```typescript
import init, { WasmNetwork } from 'skrf-wasm';

interface FileUploadEvent extends Event {
  target: HTMLInputElement;
}

async function networkExample() {
  await init();
  
  // 从文件上传读取
  const fileInput = document.getElementById('file-input') as HTMLInputElement;
  
  fileInput.addEventListener('change', async (e: FileUploadEvent) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    const content = await file.text();
    const network = WasmNetwork.fromTouchstoneContent(content, file.name);
    
    console.log(`端口数: ${network.nports}`);
    console.log(`频率点数: ${network.nfreq}`);
    
    // 获取频率数组
    const f = network.f;
    console.log(`频率范围: ${f[0] / 1e9} - ${f[f.length - 1] / 1e9} GHz`);
    
    // 检查网络特性
    console.log(`互易性: ${network.isReciprocal()}`);
    console.log(`无源性: ${network.isPassive()}`);
    
    // 获取 S21 (dB)
    const s21Db = network.getSDbAt(1, 0);
    
    // 绘制图表 (使用 Chart.js 或其他库)
    plotSParameters(f, s21Db);
  });
}

function plotSParameters(f: Float64Array, sDb: Float64Array): void {
  // 使用 Chart.js 或 D3.js 绘图
  const data = Array.from(f).map((freq, i) => ({
    x: freq / 1e9,
    y: sDb[i]
  }));
  
  console.log('Plot data:', data);
}
```

### 从 Fetch 加载

```typescript
async function loadFromUrl(url: string): Promise<WasmNetwork> {
  await init();
  
  const response = await fetch(url);
  const content = await response.text();
  
  // 从 URL 提取文件名
  const filename = url.split('/').pop() || 'unknown.s2p';
  
  return WasmNetwork.fromTouchstoneContent(content, filename);
}

// 使用示例
const network = await loadFromUrl('/data/filter.s2p');
```

---

## WasmVectorFitting 类

矢量拟合算法，用于 S 参数的有理函数逼近。

### 构造函数

```typescript
new WasmVectorFitting()
```

### 属性

| 属性 | 类型 | 可写 | 描述 |
|------|------|------|------|
| `maxIterations` | `number` | ✓ | 最大迭代次数 |
| `maxTol` | `number` | ✓ | 收敛容差 |
| `wallClockTime` | `number` | ✗ | 上次拟合耗时 (秒) |

### 方法

#### `vectorFit(network, ...)`

执行矢量拟合。

```typescript
vectorFit(
  network: WasmNetwork,
  nPolesReal?: number,        // 实极点数, 默认 2
  nPolesCmplx?: number,       // 复极点对数, 默认 2
  initPoleSpacing?: string,   // 极点分布: 'linear'|'log', 默认 'linear'
  fitConstant?: boolean,      // 拟合常数项, 默认 true
  fitProportional?: boolean   // 拟合比例项, 默认 false
): void
```

#### `getRmsError(network, i, j)`

获取 RMS 误差。

```typescript
getRmsError(network: WasmNetwork, i: number, j: number): number
```

#### `getModelOrder()`

获取模型阶数。

```typescript
getModelOrder(): number
```

#### `getModelResponse(i, j, freqs)`

获取模型响应。

```typescript
getModelResponse(
  i: number,
  j: number,
  freqs: Float64Array
): { real: Float64Array; imag: Float64Array }
```

#### `isPassive(network)`

检查模型是否被动。

```typescript
isPassive(network: WasmNetwork): boolean
```

#### `passivityTest(network)`

执行被动性测试。

```typescript
passivityTest(network: WasmNetwork): [number, number][]
```

返回违反被动性的频段数组 `[[fStart, fStop], ...]`。

#### `passivityEnforce(network, nSamples?)`

强制模型满足被动性。

```typescript
passivityEnforce(
  network: WasmNetwork,
  nSamples?: number  // 采样点数, 默认 200
): {
  success: boolean;
  iterations: number;
  historyMaxSigma: Float64Array;
}
```

#### `generateSpiceSubcircuit(network, modelName?, createReferencePins?)`

生成 SPICE 网表。

```typescript
generateSpiceSubcircuit(
  network: WasmNetwork,
  modelName?: string,           // 子电路名, 默认 's_equivalent'
  createReferencePins?: boolean // 创建参考引脚, 默认 false
): string
```

### 完整示例

```typescript
import init, { WasmNetwork, WasmVectorFitting } from 'skrf-wasm';

interface VectorFitResult {
  modelOrder: number;
  rmsErrors: Map<string, number>;
  isPassive: boolean;
  fitTime: number;
}

async function vectorFittingExample(): Promise<void> {
  await init();
  
  // 加载网络
  const response = await fetch('/data/filter.s2p');
  const content = await response.text();
  const network = WasmNetwork.fromTouchstoneContent(content, 'filter.s2p');
  
  // 创建 VectorFitting 实例
  const vf = new WasmVectorFitting();
  
  // 配置参数
  vf.maxIterations = 100;
  vf.maxTol = 1e-6;
  
  // 执行拟合
  console.log('开始矢量拟合...');
  const startTime = performance.now();
  
  vf.vectorFit(
    network,
    3,          // 3 个实极点
    5,          // 5 对复极点
    'log',      // 对数分布
    true,       // 拟合常数项
    false       // 不拟合比例项
  );
  
  const fitTime = (performance.now() - startTime) / 1000;
  console.log(`拟合完成，耗时: ${fitTime.toFixed(3)}s`);
  
  // 输出结果
  console.log(`模型阶数: ${vf.getModelOrder()}`);
  console.log(`Wall clock time: ${vf.wallClockTime.toFixed(3)}s`);
  
  // 计算 RMS 误差
  const nports = network.nports;
  for (let i = 0; i < nports; i++) {
    for (let j = 0; j < nports; j++) {
      const rms = vf.getRmsError(network, i, j);
      console.log(`S${i + 1}${j + 1} RMS Error: ${rms.toExponential(2)}`);
    }
  }
  
  // 被动性检查
  const isPassive = vf.isPassive(network);
  console.log(`模型是否被动: ${isPassive}`);
  
  if (!isPassive) {
    // 获取违反被动性的频段
    const violations = vf.passivityTest(network);
    console.log('被动性违反频段:');
    violations.forEach(([fStart, fStop]) => {
      console.log(`  ${(fStart / 1e9).toFixed(3)} - ${(fStop / 1e9).toFixed(3)} GHz`);
    });
    
    // 执行被动性强制
    console.log('执行被动性强制...');
    const result = vf.passivityEnforce(network, 200);
    console.log(`结果: ${result.success ? '成功' : '失败'}, 迭代 ${result.iterations} 次`);
  }
  
  // 获取模型响应
  const nFreqNew = 1001;
  const f = network.f;
  const fNew = new Float64Array(nFreqNew);
  for (let i = 0; i < nFreqNew; i++) {
    fNew[i] = f[0] + (f[f.length - 1] - f[0]) * i / (nFreqNew - 1);
  }
  
  const response21 = vf.getModelResponse(1, 0, fNew);
  
  // 计算 dB 值
  const s21DbModel = new Float64Array(nFreqNew);
  for (let i = 0; i < nFreqNew; i++) {
    const mag = Math.sqrt(
      response21.real[i] ** 2 + response21.imag[i] ** 2
    );
    s21DbModel[i] = 20 * Math.log10(mag);
  }
  
  // 绘制对比图
  console.log('Model S21 (dB):', s21DbModel.slice(0, 10));
  
  // 生成 SPICE 网表
  const spiceNetlist = vf.generateSpiceSubcircuit(network, 'MyFilter', false);
  console.log('SPICE Netlist Preview:');
  console.log(spiceNetlist.substring(0, 500));
  
  // 下载 SPICE 文件
  downloadFile('filter_model.sp', spiceNetlist);
}

function downloadFile(filename: string, content: string): void {
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
```

---

## React Hook 示例

```typescript
import { useState, useCallback } from 'react';
import init, { WasmNetwork, WasmVectorFitting } from 'skrf-wasm';

interface NetworkData {
  network: WasmNetwork;
  nports: number;
  nfreq: number;
  frequencies: Float64Array;
}

export function useSkrf() {
  const [initialized, setInitialized] = useState(false);
  const [network, setNetwork] = useState<NetworkData | null>(null);
  const [vf, setVf] = useState<WasmVectorFitting | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 初始化 WASM
  const initialize = useCallback(async () => {
    if (initialized) return;
    try {
      await init();
      setInitialized(true);
    } catch (e) {
      setError(`WASM 初始化失败: ${e}`);
    }
  }, [initialized]);

  // 加载 Touchstone 文件
  const loadFile = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    
    try {
      await initialize();
      const content = await file.text();
      const nw = WasmNetwork.fromTouchstoneContent(content, file.name);
      
      setNetwork({
        network: nw,
        nports: nw.nports,
        nfreq: nw.nfreq,
        frequencies: nw.f
      });
    } catch (e) {
      setError(`文件加载失败: ${e}`);
    } finally {
      setLoading(false);
    }
  }, [initialize]);

  // 执行矢量拟合
  const runVectorFit = useCallback(async (
    nPolesReal: number = 3,
    nPolesCmplx: number = 5
  ) => {
    if (!network) {
      setError('请先加载网络');
      return;
    }

    setLoading(true);
    try {
      const fitter = new WasmVectorFitting();
      fitter.vectorFit(
        network.network,
        nPolesReal,
        nPolesCmplx,
        'log',
        true,
        false
      );
      setVf(fitter);
    } catch (e) {
      setError(`矢量拟合失败: ${e}`);
    } finally {
      setLoading(false);
    }
  }, [network]);

  return {
    initialized,
    network,
    vf,
    loading,
    error,
    initialize,
    loadFile,
    runVectorFit
  };
}
```

### 使用 Hook

```tsx
import React from 'react';
import { useSkrf } from './useSkrf';

function SkrfApp() {
  const { 
    network, 
    vf, 
    loading, 
    error, 
    loadFile, 
    runVectorFit 
  } = useSkrf();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) loadFile(file);
  };

  return (
    <div>
      <h1>RF Network Analyzer</h1>
      
      <input type="file" accept=".s1p,.s2p,.s3p,.s4p" onChange={handleFileChange} />
      
      {loading && <p>Loading...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      
      {network && (
        <div>
          <h2>Network Info</h2>
          <p>Ports: {network.nports}</p>
          <p>Frequency Points: {network.nfreq}</p>
          
          <button onClick={() => runVectorFit(3, 5)}>
            Run Vector Fitting
          </button>
        </div>
      )}
      
      {vf && (
        <div>
          <h2>Fitting Result</h2>
          <p>Model Order: {vf.getModelOrder()}</p>
          <p>Fit Time: {vf.wallClockTime.toFixed(3)}s</p>
          <p>Passive: {vf.isPassive(network!.network) ? 'Yes' : 'No'}</p>
        </div>
      )}
    </div>
  );
}

export default SkrfApp;
```

---

## 错误处理

WASM 绑定会抛出 JavaScript 错误，可以用 try-catch 捕获：

```typescript
try {
  const network = WasmNetwork.fromTouchstoneContent(content, 'invalid.xyz');
} catch (e) {
  console.error('解析失败:', e);
  // 输出: "Invalid filename: expected .sNp extension (e.g., test.s2p)"
}

try {
  const order = vf.getModelOrder(); // 未拟合时调用
} catch (e) {
  console.error('模型未拟合:', e);
}
```

---

## 性能优化建议

1. **批量操作**: 尽量一次性获取所有需要的数据，减少 JS-WASM 边界调用
2. **重用实例**: `WasmVectorFitting` 实例可以重用于多次拟合
3. **Web Worker**: 对于大型网络，考虑在 Web Worker 中执行拟合
4. **Streaming**: 使用 `instantiateStreaming` 初始化 WASM 以获得更快的加载速度

```typescript
// 使用 streaming 初始化
import initSync, { WasmNetwork } from 'skrf-wasm';

// 异步流式初始化 (推荐)
await init();  // 内部使用 WebAssembly.instantiateStreaming

// 或在 Web Worker 中
self.onmessage = async (e) => {
  await init();
  const network = WasmNetwork.fromTouchstoneContent(e.data.content, e.data.filename);
  self.postMessage({ nports: network.nports, nfreq: network.nfreq });
};
```
