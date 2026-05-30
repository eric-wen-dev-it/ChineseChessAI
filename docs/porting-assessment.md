# 硬件移植评估：DGX Spark vs Ryzen AI Max（Strix Halo）

> 评估对象：本项目当前技术栈 —— C# / .NET 10（`net10.0-windows`）+ WPF UI + TorchSharp 0.105.2 + CUDA 12.8，教师蒸馏用 Pikafish。
> 目标：把 AlphaZero 自对弈训练管线迁到「AI PC / 桌面超算」级别的统一内存平台。
> 结论速览：**DGX Spark 更顺（麻烦是「ARM 上重新编译」）；Strix Halo 的 x86 让 UI/Pikafish 白捡，但训练最核心的 GPU 那一环在 AMD 上几乎没人趟过。**

---

## 0. 当前栈里与架构绑定的东西

| 组件 | 现状 | 绑定点 |
|---|---|---|
| UI | WPF，`net10.0-windows`，`<UseWPF>true` | **仅 Windows**，Linux/ARM 无法运行 |
| 深度学习 | TorchSharp 0.105.2 → **LibTorch 2.7.1 + CUDA 12.8** | 托管包架构无关；`libLibTorchSharp.so` 与 `libtorch_*.so` 绑 CPU 架构 + CUDA |
| GPU 包 | `TorchSharp-cuda-windows` 0.105.2 | 仅 win-x64 |
| 教师引擎 | `Pikafish.2026-01-02_2/Linux/*` | 全是 **x86**（avx2/avx512…），无 ARM 二进制 |
| .NET 运行时 | net10.0 | aarch64 Linux / x86 均可 |

关键认知：TorchSharp 的**托管 NuGet 包是架构无关的 IL**，真正绑架构的只有两个 `.so`：
1. `libLibTorchSharp.so` —— TorchSharp 自己的 C++ 胶水层
2. `libtorch_*.so` —— LibTorch 本体

而 **CUDA 12.8 官方构建已含 sm_120（Blackwell）**，所以 Blackwell 在版本层面天然支持，缺的只是「aarch64 编译产物」。

---

## 1. 平台对比

| 维度 | DGX Spark (GB10 Grace-Blackwell) | Strix Halo (Ryzen AI Max+ 395) |
|---|---|---|
| CPU / OS | **ARM64** Linux (DGX OS/Ubuntu) | **x86-64**，Windows/Linux |
| GPU | Blackwell，**CUDA 成熟**，版本已对齐 (2.7.1 / cu128 / sm120) | Radeon 8060S 核显（RDNA3.5，**gfx1151**），**ROCm 实验性** |
| 统一内存 | 128GB LPDDR5X | 128GB LPDDR5X |
| WPF UI | ✗ 必须剥离/重写（Avalonia 或留在 Win） | ✓ **留在 Windows 直接可用** |
| Pikafish | ✗ 需 `make ARCH=armv8` 重编 | ✓ **x86 二进制直接可用** |
| TorchSharp GPU | 重编 aarch64 二进制（**机械活**），或用现成 DGX 社区包 | **无 ROCm 支持 ⇒ 近乎自研（研究活）** |
| NPU | —（GPU 即主力） | XDNA2，仅**推理**（ONNX+VitisAI），训练用不上 |
| 本项目移植难度 | 中 —— 「重新编译」 | 高 —— 「换栈 + 骑实验性驱动」 |

---

## 2. DGX Spark 三套 TorchSharp-on-ARM-CUDA 方案

前提：需要在 aarch64 上把 `libLibTorchSharp.so` 按 0.105.2 重编，并链接 aarch64 版 libtorch 2.7.1（cu128）。aarch64+cu128 的 libtorch 2.7.1 可从 `download.pytorch.org/whl/cu128` 的 sbsa wheel 里取（`torch/lib`），或用 NGC 容器预装的。版本正好对齐，ABI 可匹配。

### 方案 A —— NGC 容器里从源码编胶水层（改动最小，推荐）
1. 在 aarch64 CUDA 容器（`nvcr.io/nvidia/pytorch`）里装 aarch64 libtorch 2.7.1/cu128。
2. `git checkout v0.105.2`，用 `-DCMAKE_PREFIX_PATH=<libtorch>` 编 `src/Native`。
3. **唯一的坑**（GitHub dotnet/TorchSharp#1516）：构建脚本会无视指定路径去 `src/Redist/libtorch-cpu` **自动下载 x64 libtorch** 导致架构不匹配 —— 必须**禁掉 Redist 自动下载**并强指向本地 aarch64 libtorch。
4. .NET 项目**删 `TorchSharp-cuda-windows`，只留托管 `TorchSharp` 0.105.2**，`.so` 放 `LD_LIBRARY_PATH` 或程序目录。
- 好处：C# 训练代码几乎不动，API 仍是 0.105.2。

### 方案 B —— 现成社区包 `FAkka.TorchSharp.DGX`（最快验证可行性）
- 专为 **DGX Blackwell/GB10、linux-arm64** 打的包（~135MB，动态链接 NGC 容器里的 PyTorch）。
- 代价：对应 NGC `pytorch:25.01` + 它 fork 的托管版本 26.1.0，**API 面和 0.105.2 不同**，代码要跟着改；第三方 fork，长期维护性存疑。
- 定位：容器里做一次「GPU 能否跑通」的 spike。

### 方案 C —— 不在 C# 里跑 GPU LibTorch（最稳、性能最好）
- C# 只留 MCTS/自对弈/规则（CPU）；**训练用 Python PyTorch**（DGX Spark 上 aarch64+Blackwell 是一等公民，`pip install torch --index-url .../cu128` 即可）。
- 自对弈推理：模型导出 **ONNX**，用 ONNX Runtime（linux-arm64 CUDA/TensorRT EP）或 **TensorRT**（Blackwell 上性能最优）。
- 代价：训练循环用 Python 重写，两边靠权重文件（`.pt`/safetensors）交换。

---

## 3. Strix Halo（AMD）专属注意

- **无 CUDA**，GPU 走 ROCm/HIP，而 **TorchSharp 完全没有 ROCm 包**，也没有 DGX 那种现成社区包 ⇒ 想在 C# 用上 AMD GPU，得拿 ROCm 版 libtorch（**仅 linux-x64**）自己重编胶水层，比 ARM+CUDA 更难（Windows 侧只有 Python wheel，没有干净的 C++ SDK）。
- **gfx1151 的 ROCm 支持还很嫩**：2025 年底才进实验性，官方支持矩阵未正式列（靠 gfx1100 ISA 兼容）；ROCm 7.2（2026-03）才给 Strix Halo 上 Windows。**训练侧**尤其粗糙：分布式 collective 未做全、`torchao`/`bitsandbytes` import 即崩、部分 kernel 受 `hipMemcpyWithStream` 拖累性能不达标。
- XDNA2 NPU 只能**推理**，对训练主循环没用。
- 现实可行路径几乎只有**方案 C**（Python-PyTorch-ROCm 训练 + C# 逻辑 + ONNX 推理），且要忍受核显 ROCm 的实验状态。此时 Strix Halo 的价值在「x86 + 128GB + 强 CPU + 本地推理」，而非替代现有 CUDA 训练管线。

---

## 4. 建议

- **想少改代码、继续用 C# 训练** → **DGX Spark + 方案 A**，主要工作量是搞定「禁 Redist 下载 + 指向 aarch64 libtorch」这个编译坑。
- **先花半天验证** → DGX Spark 上跑方案 B spike。
- **愿意重构、压榨硬件** → 方案 C（两平台通用；Strix Halo 实际上只能走这条）。
- 单从「本项目移植顺滑度」看：**DGX Spark 明显优于 Strix Halo**。Strix Halo 只有在「愿意把训练整块迁到 Python + 接受核显 ROCm 实验状态」时才谈得上可行。

---

## 参考

- TorchSharp #1516 — build native pack for arm64 ubuntu: https://github.com/dotnet/TorchSharp/issues/1516
- TorchSharp 0.105.2（libtorch 2.7.1 / CUDA 12.8）: https://www.nuget.org/packages/TorchSharp/0.105.2
- FAkka.TorchSharp.DGX（linux-arm64, Blackwell/GB10）: https://libraries.io/nuget/FAkka.TorchSharp.DGX
- PyTorch aarch64 CUDA 12.8 / sm_120 Blackwell: https://github.com/pytorch/pytorch/pull/146378
- DGX Spark PyTorch & CUDA guide: https://github.com/martimramos/dgx-spark-ml-guide
- PyTorch + ROCm 7 on Strix Halo (gfx1151), Windows: https://medium.com/@GenerationAI/pytorch-with-rocm-7-for-windows-on-amd-ryzen-ai-max-395-strix-halo-radeon-8060s-gfx1151-1ba069edc2c4
- ROCm/TheRock — gfx1151 PyTorch wheels: https://github.com/ROCm/TheRock/discussions/655
- Fine-tuning LLMs on Strix Halo（训练侧限制）: https://www.promptinjection.net/p/how-to-fine-tune-llms-on-amd-strix-halo-ryzen-ai-max-395-sft-lora
- pytorch/pytorch #171687 — gfx1151 训练性能: https://github.com/pytorch/pytorch/issues/171687
