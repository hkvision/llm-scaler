@echo off
REM build_moe_int4.bat — build the int4 MoE ESIMD kernel (moe_int4_ops) for Arc B390 (PTL/XE3)
REM Follows skill development/cross-os/build-esimd-kernels: manual PATH/INCLUDE/LIB
REM (NEVER vcvarsall.bat — it overflows PATH). JIT SYCL build (no -device bmg).
REM Output: python\custom_esimd_kernels_vllm\moe_int4_ops.cp311-win_amd64.pyd  (in-place)

setlocal EnableDelayedExpansion

set "SRC=C:\Users\kai\kai\llm-scaler\vllm\custom-esimd-kernels-vllm"
set "CONDA_ENV=C:\Users\kai\miniforge3\envs\kai-vllm"
set "PYTHON=%CONDA_ENV%\python.exe"
set "CMPLR_ROOT=C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3"
set "LOG=C:\Users\kai\kai\perf\build_moe_int4.log"

REM --- auto-detect VS2022 (found: Community in Program Files) ---
set "VS_ROOT="
for %%E in (Community Professional Enterprise BuildTools) do (
    if not defined VS_ROOT if exist "C:\Program Files\Microsoft Visual Studio\2022\%%E\VC\Tools\MSVC" set "VS_ROOT=C:\Program Files\Microsoft Visual Studio\2022\%%E"
    if not defined VS_ROOT if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\%%E\VC\Tools\MSVC" set "VS_ROOT=C:\Program Files (x86)\Microsoft Visual Studio\2022\%%E"
)
set "MSVC_VER="
for /f "tokens=*" %%V in ('dir /b /o-n "%VS_ROOT%\VC\Tools\MSVC" 2^>nul') do if not defined MSVC_VER set "MSVC_VER=%%V"
set "MSVC_ROOT=%VS_ROOT%\VC\Tools\MSVC\%MSVC_VER%"

REM --- auto-detect Windows SDK (10.* only) ---
set "WINSDK_ROOT=C:\Program Files (x86)\Windows Kits\10"
set "WINSDK_VER="
for /f "tokens=*" %%V in ('dir /b /o-n "%WINSDK_ROOT%\Include" 2^>nul ^| findstr /b "10\."') do if not defined WINSDK_VER set "WINSDK_VER=%%V"

REM --- auto-detect clang version inside oneAPI ---
set "CLANG_VER="
for /f "tokens=*" %%V in ('dir /b /o-n "%CMPLR_ROOT%\lib\clang" 2^>nul') do if not defined CLANG_VER set "CLANG_VER=%%V"

REM --- build env from scratch (short PATH; no vcvarsall/setvars) ---
set "PATH=%CMPLR_ROOT%\bin;%CMPLR_ROOT%\bin\compiler;%CONDA_ENV%;%CONDA_ENV%\Scripts;%CONDA_ENV%\Library\bin"
set "PATH=%PATH%;%MSVC_ROOT%\bin\HostX64\x64"
set "PATH=%PATH%;%WINSDK_ROOT%\bin\%WINSDK_VER%\x64"
set "PATH=%PATH%;C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem"

set "INCLUDE=%CMPLR_ROOT%\include;%CMPLR_ROOT%\include\sycl"
set "INCLUDE=%INCLUDE%;%MSVC_ROOT%\include"
set "INCLUDE=%INCLUDE%;%WINSDK_ROOT%\Include\%WINSDK_VER%\ucrt"
set "INCLUDE=%INCLUDE%;%WINSDK_ROOT%\Include\%WINSDK_VER%\um"
set "INCLUDE=%INCLUDE%;%WINSDK_ROOT%\Include\%WINSDK_VER%\shared"

set "LIB=%CMPLR_ROOT%\lib\clang\%CLANG_VER%\lib\windows;%CMPLR_ROOT%\lib"
set "LIB=%LIB%;%MSVC_ROOT%\lib\x64"
set "LIB=%LIB%;%WINSDK_ROOT%\Lib\%WINSDK_VER%\ucrt\x64"
set "LIB=%LIB%;%WINSDK_ROOT%\Lib\%WINSDK_VER%\um\x64"

set VLLM_TARGET_DEVICE=xpu
set CXX=icx-cl
set CC=icx-cl

echo ============ build env ============ > "%LOG%"
echo VS_ROOT=%VS_ROOT%  MSVC_VER=%MSVC_VER%  WINSDK_VER=%WINSDK_VER%  CLANG_VER=%CLANG_VER% >> "%LOG%"
echo PYTHON=%PYTHON% >> "%LOG%"
echo. >> "%LOG%"

cd /d "%SRC%"
echo [build] python setup_moe_int4_ptl.py build_ext --inplace >> "%LOG%"
"%PYTHON%" setup_moe_int4_ptl.py build_ext --inplace >> "%LOG%" 2>&1
echo [build exit code] %ERRORLEVEL% >> "%LOG%"

echo. >> "%LOG%"
echo ============ smoke test (load pyd directly, bypass broken package __init__) ============ >> "%LOG%"
"%PYTHON%" -c "import glob,importlib.util,torch; p=glob.glob(r'%SRC%\python\custom_esimd_kernels_vllm\moe_int4_ops*.pyd')[0]; s=importlib.util.spec_from_file_location('moe_int4_ops',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('LOADED', p); print('ops:', [n for n in dir(torch.ops.moe_int4_ops)])" >> "%LOG%" 2>&1
echo [smoke exit code] %ERRORLEVEL% >> "%LOG%"
echo Done. Log: %LOG%
endlocal
