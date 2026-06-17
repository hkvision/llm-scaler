# DiffusionGemma ESIMD int4 MoE 内核 — Windows/PTL 编译问题记录

> 目的:在 **Arc B390 (Panther Lake iGPU, PTL/XE3)** + Windows 上,把 Linux 团队的
> ESIMD int4 MoE 内核 `moe_int4_ops`(`csrc/moe_batch/moe_int4.sycl`)编译出来,
> 用于替换 vLLM `sym_int4.py` 里 ~87ms/层的 PyTorch MoE 慢回退。
> 本文记录编译过程中踩到的全部问题、根因、改法,供与内核作者/同事对齐。
>
> 状态:**编译通过 + .pyd 可加载,导出 18 个内核函数。** 集成尚未开始。

---

## 0. 环境

| 组件 | 版本 |
|---|---|
| GPU | Arc B390 = **PTL/XE3 (XeLPG iGPU)**,**无 dpas2**,共享主机内存,TDR ~2s |
| 编译器 | Intel oneAPI **2025.3**(icx,内置 clang **21**) |
| MSVC | VS2022 Community **14.44.35207** |
| Windows SDK | 10.0.26100 |
| PyTorch | 2.11.0a0+git426d9ef(XPU,从 wheel 装,ABI 已匹配) |
| Python | 3.11 / conda env `kai-vllm` |
| 构建后端 | 仓库自带 `esimd_build_extention.py`(ninja + setuptools),**非** torch 自带 SyclExtension |

构建命令:
```
cmd /c "C:\Users\kai\kai\perf\build_moe_int4.bat"
# → setup_moe_int4_ptl.py build_ext --inplace
# 日志:perf\build_moe_int4.log
```

> 关键前提(沿用内部 skill `build-esimd-kernels`):**手动拼 PATH/INCLUDE/LIB,绝不调
> `vcvarsall.bat`**(会导致 PATH overflow);用 `esimd_build_extention` 而不是 torch
> 自带 SyclExtension(后者会加 `-fsycl-host-compiler`,混合 MSVC+clang 内联头,问题更多)。

---

## 问题 1 — `.sycl` 源文件在 Windows 下被路由到 `cl`,链接报 LNK1181

**现象**
```
LNK1181: cannot open input file '...moe_int4.obj'
```
`.sycl` 没被识别成 SYCL 源,落到了 MSVC `cl` 编译规则。

**根因**
`esimd_build_extention.py` 的 Windows 路径 `win_wrap_ninja_compile()` 把
`with_sycl=False` 写死,只有 Unix 路径 (`unix_wrap_ninja_compile`) 正确检测 SYCL。

**改法**(`esimd_build_extention.py` ~L989,即内部 skill 记的 "Bug 2")
在 Windows 路径里加 `with_sycl` 检测,镜像 Unix 路径构造 sycl 编译规则,让
`_write_ninja_file` 用 `$sycl`(icx)发出 `sycl_compile` rule。**关键:不加
`-fsycl-host-compiler`。**

---

## 问题 2 — ninja 报 `expected build command name`

**现象**
```
ninja: error: build.ninja:NN: expected build command name
```

**根因**
`sycl_devlink` 这条 build 行里的 Windows 盘符冒号(`C:`)没转义。ninja 把 `:` 当
路径分隔符。(普通 compile build 行已转义,devlink 行漏了。)

**改法**(`esimd_build_extention.py` ~L2897,"Bug 3")
devlink 的 out/in 路径做 `C:` → `C$:`、空格 → `$ ` 转义。

---

## 问题 3 ★ — `conflicting types for '_mm_div_epi8'`(本次最关键,且曾被误诊)

**现象**
```
C:\...\MSVC\14.44.35207\include\immintrin.h(2254,16):
    error: conflicting types for '_mm_div_epi8'
    note: '_mm_div_epi8' is a builtin with type '... long long (...) noexcept'
```
几十个 SVML intrinsic(`_mm_div_epi8/16/32`、`_mm_rem_*`、`_mm256_*`…)全部冲突。
include 链:`torch/extension.h` → … → `wchar.h` → `intrin.h` → `immintrin.h`。

### 上个 session 的误诊(❌ 错误,已推翻)
当时判断为「clang-21 与 VS 14.44 的 SVML 头根本不兼容,需要**管理员权限改系统
immintrin.h / zmmintrin.h** 给 SVML 块加 guard」,并尝试了 shadow header 目录
(`perf/patched_include`)。结果 shadow 方案引出新的 `__v2df/__v4df` 类型不匹配错误,
彻底卡死。**这个方向是错的。**

### 真正根因(✅ 用最小复现证明)
1. `#include <immintrin.h>` **单独编译完全没问题**(任何 icx/icx-cl 驱动+flag 组合,
   exit 0)。→ 不是工具链本身不兼容。
2. 冲突只在 **MSVC 的 immintrin.h 被排在 clang-21 自己的 builtin intrinsic 头之前**
   时出现。
3. `esimd_build_extention`(经 distutils 的 `MSVCCompiler.initialize()`)把 MSVC
   VC-Tools 和 Windows-SDK 的 include 目录当成**普通 `-I`(高优先级)** 注入。这会
   **盖过 clang 的 builtin 头**。于是 MSVC immintrin.h 里的
   `extern __m128i _mm_div_epi8(__m128i, __m128i)` 声明,和 clang-21 内建的同名 SVML
   **builtin** 撞了 → "conflicting types"。

**决定性 A/B 实验**(同样的 MSVC/SDK 目录,只改传入方式,其余 flag 完全相同):

| 方式 | 结果 |
|---|---|
| 用 `-I<dir>` 传 MSVC/SDK 头(= esimd 默认行为) | **FAIL** `conflicting types for '_mm_div_epi8'` |
| 用 `-imsvc <dir>` 传(clang 的 MSVC-system include) | **PASS**,完整 device 编译 exit 0 |

复现脚本:`perf/try_order.bat` / `try_order.log`(R1=`-I` 失败,R2=`-imsvc` 通过)。

### 改法(`esimd_build_extention.py` ~L1003,本次新增,skill 未覆盖)
在 `win_wrap_ninja_compile` 的 `with_sycl:` 分支里,构造 `sycl_cflags` 后把指向
MSVC / Windows-SDK 的 `-I` 改写成 `-imsvc`:
```python
def _to_imsvc(flag):
    if flag.startswith('-I') and (
        'Microsoft Visual Studio' in flag or 'Windows Kits' in flag):
        return '-imsvc' + flag[2:]
    return flag
sycl_cflags = [_to_imsvc(f) for f in sycl_cflags]
```
`-imsvc` 让 MSVC 头排在 clang builtin 头**之后** → SVML 声明与 builtin 不再冲突。
**不需要改任何系统头文件。**

> ⚠️ 给同事的提醒:这个问题大概率只在 **较新的 MSVC(14.44,immintrin.h 里无条件声明
> 了 SVML)** + oneAPI clang-21 组合下出现。旧版 MSVC 的 immintrin.h 没有这段 SVML 声明,
> 所以不会撞 —— 这也解释了为什么 Linux 端 / 旧环境从没碰到、文档里也没写。
> 如果你们的环境复现不出,先 `echo %VS 版本%` 对一下 MSVC 版本。

---

## 问题 4 — `clang-offload-bundler: 'mtl,mtl-h,bmg,dg2,arl-h,lnl-m,ptl': no such file`

**现象**(问题 3 修好后,内核 3000+ 行全部编过,卡在 AOT 打包阶段)
```
clang-offload-bundler: error: 'mtl,mtl-h,bmg,dg2,arl-h,lnl-m,ptl': no such file or directory
icx: error: clang-offload-bundler command failed with exit code 1
```

**根因**
`esimd_build_extention` 默认 `_COMMON_SYCL_FLAGS` 写死 `-fsycl-targets=spir64_gen,spir64`
(AOT),而 `_SYCL_DLINK_FLAGS` 又加了 `-Xs "-device {torch.xpu.get_arch_list()}"`。
`get_arch_list()` 返回机器上**所有** GPU arch(`mtl,...,ptl`),这串被 offload-bundler
当成文件名了。而且 PTL/XE3(XeLPG,无 dpas2)本来就**不该走 AOT**。

**改法**(`esimd_build_extention.py` L287 / L305)
- `_COMMON_SYCL_FLAGS`: `-fsycl-targets=spir64_gen,spir64` → **`-fsycl-targets=spir64`**(纯 JIT SPIR-V,运行时再 JIT)。
- `_SYCL_DLINK_FLAGS`: 去掉 `-Xs "-device {arch_list}"`(那是 AOT 后端选项,JIT 下无意义且报错)。

> 与内部 skill "Bug 4" 的关系:skill Bug 4 针对的是 **AOT-BMG** 场景(把 per-extension
> 的 `-device bmg` 传给 dlink)。我们是 **JIT-PTL** 场景 —— 同一个全局 arch list 惹的祸,
> 但按 PTL=JIT 的原则直接把 AOT 默认值去掉,而不是用 Bug 4 的 per-extension 传递。

---

## 附:其它已应用的小改动

- **`moe_int4.sycl` L3266**:38 个 kernel 里唯一一个 `parallel_for` 没命名 →
  补成 `parallel_for<class MoeInt4NMajorGemmEsimd>`。命名 kernel 是 SYCL 要求(疑似上游遗漏)。

---

## 改动文件清单

| 文件 | 改动 |
|---|---|
| `vllm/custom-esimd-kernels-vllm/esimd_build_extention.py` | 4 处 `PATCH(kai)`:L287 JIT spir64、L305 去 dlink 的 -Xs device、L992 Bug2 sycl 路由、L1003 **-imsvc 改写** + L2897 Bug3 盘符转义 |
| `vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl` | L3266 kernel 命名 |
| `vllm/custom-esimd-kernels-vllm/setup_moe_int4_ptl.py` | 新增,JIT 版构建脚本(已移除 patched_include 残留) |
| `perf/build_moe_int4.bat` | 新增,构建启动器 + load .pyd smoke test |

## 作废 / 别再试的方向

- ❌ 改系统 `immintrin.h` / `zmmintrin.h`(完全不需要)
- ❌ `perf/patched_include` shadow 头目录(任何变体 —— `__v2df/__v4df` 报错的来源)
- ❌ `-fms-compatibility-version=`(无效,MSVC 的 SVML 块是无条件声明)

## 复现脚本(都在 `perf/`)

| 脚本 | 用途 | 结论 |
|---|---|---|
| `try_imm_fix.bat` / `try_imm_fix2.bat` | `#include <immintrin.h>` 单独编,各驱动/flag | 全 PASS → 非工具链问题 |
| `try_inc_bisect.bat` | 二分哪个头触发(esimd / torch / cmath) | torch 头链触发 |
| `try_order.bat` | **`-I` vs `-imsvc` A/B** | R1(-I)FAIL / R2(-imsvc)PASS |
| `build_moe_int4.bat` | 完整构建 | exit 0,产出 .pyd |

## 最终结果

```
[build exit code] 0
[smoke exit code] 0
LOADED ...\moe_int4_ops.cp311-win_amd64.pyd
module attrs: moe_forward_full_int4, moe_router_forward_int4, moe_gemm_int4_nmajor,
  moe_silu_mul_int4, moe_route_precompute_int4, moe_route_gather_int4, moe_topk_int4,
  moe_router_topk_int4, moe_shared_expert_forward_int4_nmajor, moe_forward_cutlass_nmajor_int4_full,
  moe_tiny_cutlass_nmajor_int4_up/down, moe_tiny_fp16_shared_up/finalize, ... (共 18 个)
```
> 注:这些是 **pybind11 模块函数**(`moe_int4_ops.moe_forward_full_int4(...)`),不是注册到
> `torch.ops.moe_int4_ops` 命名空间。`python/.../ops.py` 的 wrapper 就是按模块函数调用的。

## 下一步(未做)

把内核接进 `sym_int4.py` 的 `_moe_torch_int4_forward`,替换 128-expert Python 循环;
需要处理权重布局映射(Q4_0:int32 packed nibble + fp16 group scale gs=128;内核读 IPEX
`[E, K_packed, N]` 布局,见 `moe_int4.sycl` 头部注释)。注意运行时 gotchas:
`lsc_load_2d<bf16>` 在 PTL 坏、首次 JIT 有 ~2s TDR。然后对着 ~87ms/层基线重新 profile。
