#ifndef _CUDA_KERNELS_H
#define _CUDA_KERNELS_H

__global__ void sim_kernel_shfl(double *, double *, size_t, size_t, double, double, double);
__global__ void sim_kernel_tiled(double *, double *, size_t, size_t, double, double, double);
__global__ void sim_kernel_naive(double *, double *, size_t, size_t, double, double, double);
__global__ void block_min_max_kernel(double *, double *, size_t, double *, double *);
__global__ void grayscale_kernel(double *, unsigned char *, size_t, double, double);

#endif // _CUDA_KERNEL_H defined
