# SBPF MLIR 项目安装指南

## 项目结构

```
src/
├── frontend/           # Rust ELF 解析器 (已完成)
├── mlir-backend/       # Melior Rust 后端
└── mlir-dialect/       # C++ SBPF Dialect (TableGen)
    ├── include/sbpf/
    │   ├── SbpfDialect.td   # 方言定义
    │   ├── SbpfOps.td       # 操作定义
    │   ├── SbpfDialect.h
    │   └── SbpfOps.h
    ├── lib/
    │   ├── SbpfDialect.cpp
    │   └── SbpfOps.cpp
    ├── tools/
    │   └── sbpf-opt.cpp     # 优化器驱动
    └── test/
        └── basic.mlir       # 测试用例
```

## 1. 构建 SBPF Dialect (C++)

你已经有了 LLVM 22 在 `~/Documents/Archive/code/MLIR/base/llvm-project/build`。

```bash
cd ~/Documents/My-data/Demo/src/mlir-dialect

# 创建构建目录
mkdir -p build && cd build

# 配置 (指向你的 LLVM 22 构建)
cmake .. \
  -DLLVM_DIR=~/Documents/Archive/code/MLIR/base/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=~/Documents/Archive/code/MLIR/base/llvm-project/build/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Debug

# 编译
cmake --build . --target sbpf-opt

# 测试
./sbpf-opt ../test/basic.mlir
```
