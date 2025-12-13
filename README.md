# cuSharedContext

A demonstration project showcasing CUDA context sharing across multiple dynamically loaded shared libraries (DLLs/SOs). This project illustrates how CUDA kernels can be safely used across different shared libraries that are loaded into the same process, sharing a common CUDA context.

## Overview

This project demonstrates a common pattern in CUDA application development where multiple shared libraries need to use GPU compute capabilities independently while sharing the same CUDA context. The architecture includes:

- **Common Library**: Provides shared CUDA array management and kernel utilities
- **Shared Library A & B**: Two independent shared libraries that each implement CUDA kernels
- **Driver Application**: A CLI application that loads and uses both shared libraries

The key architectural feature is that all libraries can allocate GPU memory, launch kernels, and interoperate seamlessly because they share the same CUDA context.

## Features

- CUDA Unified Memory management through a custom `Array` template class
- Multiple shared libraries with independent CUDA kernel implementations
- Cross-library GPU array operations and data sharing
- CMake-based build system with CUDA support
- CLI interface using CLI11 for easy testing
- Doxygen documentation support
- GitHub Actions CI/CD pipeline

## Requirements

- CUDA Toolkit (11.2+)
- CMake 3.20+
- C++11 compatible compiler
- NVIDIA GPU with CUDA support
- (Optional) Doxygen for documentation generation
- (Optional) ccache for faster rebuilds

## Building

### Basic Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Build with Documentation

```bash
mkdir build && cd build
cmake -DBUILD_DOCS=ON ..
cmake --build .
cmake --build . --target doxygen
```

### Build Configuration Options

- `BUILD_DOCS`: Enable Doxygen documentation generation (default: OFF)

## Usage

The driver application provides a CLI interface to test the CUDA shared library functionality:

```bash
./build/src/driverApp/driverApp [OPTIONS]
```

### Command Line Options

- `-s, --size <N>`: Array size in elements (default: 1000)
- `-n, --iterations <N>`: Number of iterations (default: 10)
- `-v, --verbose`: Enable verbose output

### Example

```bash
# Run with default parameters
./build/src/driverApp/driverApp

# Run with custom array size and iterations
./build/src/driverApp/driverApp --size 10000 --iterations 100

# Run with verbose output
./build/src/driverApp/driverApp -v
```

## Project Structure

```
cudaDLLContext/
├── src/
│   ├── common/              # Common CUDA utilities and array management
│   │   ├── include/common/
│   │   │   ├── Array.hpp    # CUDA unified memory array wrapper
│   │   │   ├── kernels.hpp  # Common CUDA kernels
│   │   │   └── export.hpp   # Export macros
│   │   └── src/
│   ├── sharedLibA/          # First shared library with CUDA kernels
│   │   ├── include/sharedLibA/
│   │   │   └── sharedLibA.hpp
│   │   └── src/
│   ├── sharedLibB/          # Second shared library with CUDA kernels
│   │   ├── include/sharedLibB/
│   │   │   └── sharedLibB.hpp
│   │   └── src/
│   └── driverApp/           # Main driver application
│       ├── main.cpp         # CLI entry point
│       └── CLI11.hpp        # Command line parser
├── docs/                    # Doxygen documentation configuration
├── CMakeLists.txt          # Main CMake configuration
└── README.md               # This file
```

## How It Works

The application demonstrates CUDA context sharing through a series of operations:

1. **Array Creation**: Arrays are created in different shared libraries using CUDA unified memory
2. **Cross-Library Processing**: Arrays created in one library can be processed by kernels in another library
3. **Kernel Combination**: Multiple libraries can perform kernel operations on the same data
4. **Verification**: Results are compared across libraries to ensure correctness

### Example Workflow

```cpp
// Create array in sharedLibA
auto aArr = sharedLibA::createArray(size);
sharedLibA::inPlaceprocess(aArr);

// Process the same array in sharedLibB
sharedLibB::inPlaceprocess(aArr);

// Combine results from both libraries
auto combo = driverApp::combine(aArr, bArr);
```

## Key Components

### Array Class

The `Array<T>` template class provides RAII-style management of CUDA unified memory:

- Automatic memory allocation/deallocation
- Move semantics for efficient transfers
- GPU kernel integration
- Type-safe element access

### Shared Libraries

Each shared library (A and B) provides:

- `createArray()`: Allocate a new CUDA array
- `inPlaceprocess()`: Process array data with CUDA kernels
- `combine()`: Combine two arrays using GPU computation

## Development

### Code Formatting

The project includes automatic code formatting with clang-format:

```bash
cmake --build build --target fmt
```

### Continuous Integration

GitHub Actions automatically builds the project and generates documentation on every push to main or pull request.

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2022 Chris Petrovitch

## Contributing

Contributions are welcome! Please ensure:

1. Code follows the project's clang-format style
2. All builds pass without warnings
3. Documentation is updated for new features

## Troubleshooting

### CUDA Not Found

Ensure the CUDA Toolkit is installed and `nvcc` is in your PATH:

```bash
nvcc --version
```

### Build Errors

- Verify CMake version is 3.20 or higher: `cmake --version`
- Ensure your GPU supports the CUDA compute capability being used
- Check that CUDA libraries are properly installed

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CMake CUDA Support](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#cuda)
- [CUDA Unified Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)
