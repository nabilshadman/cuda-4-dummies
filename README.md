# CUDA 4 Dummies

> A comprehensive introduction to GPU computing with NVIDIA CUDA — from basic concepts to practical scientific applications.

[![Course](https://img.shields.io/badge/Course-Oct%202025-blue)](https://events.asc.ac.at/event/208/)
[![Level](https://img.shields.io/badge/Level-Beginner%20to%20Intermediate-green)]()
[![License](https://img.shields.io/badge/License-Educational-orange)]()

## Overview

This repository contains all lecture materials, hands-on exercises, and tutorial files from the **CUDA 4 Dummies** course, a 2-day intensive training program organized by the Austrian Scientific Computing (ASC) Research Center at TU Wien. The course provides a systematic, step-by-step introduction to GPU programming from the perspective of newcomers, focusing on practical skills for scientific computing.

## Course Information

- **Dates**: October 22-23, 2025
- **Format**: Live online (Zoom)
- **Duration**: 09:00 – 17:00 CEST daily
- **Target Audience**: Researchers, developers, and students in academia, industry, and public administration
- **Prerequisites**: C/C++ programming experience and Linux command-line familiarity
- **Platform**: Hands-on labs conducted on VSC-5 supercomputer

## What You'll Learn

### Day 1: Fundamentals
- GPU architecture and CUDA programming model
- Kernel execution configurations and thread hierarchies
- Memory management (unified memory, device/host transfers)
- Basic optimization techniques
- **Tutorials**: Monte Carlo π estimation, Coulomb interaction calculations

### Day 2: Advanced Topics
- CUDA SDK exploration and best practices
- High-performance libraries (cuBLAS, cuSolver)
- Profiling and performance optimization with Nsight tools
- CUDA streams and concurrent execution
- Numerical accuracy strategies on consumer-grade GPUs
- **Tutorials**: Matrix operations, eigenvalue problems, stream optimization

## Repository Structure

```
.
├── 01_Introduction_to_GPU_computing_with_CUDA/
│   ├── notes-l1.pdf                          # Lecture slides
│   ├── notes-ho1.pdf                         # Hands-on guide
│   └── hands-on-1/                           # Exercise files
│
├── 02_Memory_hierarchies_in_CPU_GPU_architectures/
│   ├── notes-l2.pdf
│   ├── notes-ho2.pdf
│   └── hands-on-2/
│
├── 03_Tutorial_1/
│   ├── notes-t1.pdf
│   └── tutorial-1/                           # Pi & Coulomb examples
│       ├── pi_v0.c → pi_solution_v4.cu
│       └── coulomb_v0.c → coulomb_solution_v1.cu
│
├── 04_CUDA_SDK_basic_concepts/
│   ├── notes-l3.pdf
│   ├── notes-ho3.pdf
│   └── hands-on-3/                           # SDK examples
│
├── 05_CUDA_SDK_libraries_numerical_accuracy/
│   ├── notes-l4.pdf
│   ├── notes-ho4.pdf
│   └── hands-on-4/                           # cuSolver examples
│
└── 06_Tutorial_2/
    ├── notes-t2.pdf
    └── tutorial-2/                           # Advanced exercises
        ├── mmm_example_[1-3].cu              # Matrix multiplication
        └── stream_test*.cu                   # Concurrency examples
```

## Key Examples

### Progressive Learning Approach
Each concept is introduced through multiple versions showing iterative optimization:

- **Matrix Addition**: Single thread block → Multiple thread blocks → Unified memory
- **Pi Calculation** (5 versions): CPU → Basic GPU → Random generation on GPU → Thread block reductions → Local memory optimization
- **Matrix Multiplication** (3 versions): Naive implementation → Shared memory (2.9× speedup)
- **Coulomb Interactions**: CPU double-loop → GPU parallelization with 4,383 atoms

### Performance Highlights
- Unified memory bandwidth: 40 GB/s → 1,341 GB/s (with prefetching)
- Matrix multiplication: 2.9× speedup using shared memory
- PCIe bandwidth: ~25 GB/s (pinned memory) vs 1,555 GB/s (device memory)

## Getting Started

### Prerequisites
```bash
# Required
- NVIDIA CUDA Toolkit (≥11.8)
- C/C++ compiler (GCC/NVCC)
- Linux environment

# Optional but recommended
- CMake (≥3.29)
- NVIDIA Nsight Compute/Systems for profiling
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/cuda-4-dummies.git
cd cuda-4-dummies

# Compile a basic example
cd 01_Introduction_to_GPU_computing_with_CUDA/hands-on-1
nvcc single_thread_block_matrix_addition.cu -o matrix_add
./matrix_add

# Profile your first CUDA program
nsys nvprof ./matrix_add
```

### Example: Vector Addition
```cuda
__global__ void VecAdd(float *A, float *B, float *C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
    // Unified memory allocation
    cudaMallocManaged(&A, N * sizeof(float));
    
    // Kernel launch
    VecAdd<<<1, N>>>(A, B, C);
    cudaDeviceSynchronize();
    
    cudaFree(A);
}
```

## Course Highlights

### Hands-On Exercises
- **MM Challenges**: Timed optimization challenges during lectures
- **Individual Support**: Dedicated time slots for personalized assistance
- **Real Scientific Problems**: Protein electrostatics, numerical methods, eigenvalue problems

### Tools & Libraries Covered
- **Profiling**: `nsys`, `ncu`, `nvidia-smi`
- **Libraries**: cuBLAS, cuSolver, cuFFT, cuRAND
- **Debugging**: `cuda-gdb`, `assert()` in kernels
- **Compilation**: `nvcc`, `cmake`, mixed Fortran/C/CUDA

### Best Practices Emphasized
✓ Use unified memory on Pascal+ GPUs  
✓ Prefer pinned memory for host-device transfers  
✓ Leverage shared memory for frequently accessed data  
✓ Profile early and often  
✓ Watch for thread divergence and load imbalance  
✓ Understand memory access patterns  

## Instructors

- **Siegfried Höfinger** – Lecturer, ASC Research Center, TU Wien
- **Atul Singh** – Teaching Assistant, ASC Research Center, TU Wien
- **Ivan Vialov** – Teaching Assistant, ASC Research Center, TU Wien

## Additional Resources

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [VSC-5 Documentation](https://asc.ac.at/systems/vsc-5/)
- [EuroCC Austria](https://www.eurocc-austria.at/)

## Citation

If you use these materials in your work, please cite:

```bibtex
@course{cuda4dummies2025,
  title        = {CUDA 4 Dummies},
  author       = {Höfinger, Siegfried and Singh, Atul and Vialov, Ivan},
  organization = {ASC Research Center, TU Wien},
  year         = {2025},
  month        = {October},
  url          = {https://events.asc.ac.at/event/208/}
}
```

## Acknowledgments

This course is partially funded by the **EuroCC 2 project** under grant agreement No 101101903 from the European High-Performance Computing Joint Undertaking (JU), with support from the Digital Europe Programme and multiple European countries.

Additional funding provided by Austrian federal ministries BMBWF and BMK.

## License

These materials are provided for educational purposes. Please contact the course organizers for usage permissions beyond personal learning.

## Contact

For questions about the course content:
- **Email**: training@asc.ac.at
- **Course Website**: https://events.asc.ac.at/event/208/

---

*Made with ❤️ for the GPU computing community*
