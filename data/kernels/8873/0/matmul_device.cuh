#ifndef CUDA_MATMUL_CUH
#define CUDA_MATMUL_CUH

enum MatmulImplementation { NAIVE, CACHE, SHARED };

void cudaMatmul(float *d_a, float *d_b, float *d_c, int n, MatmulImplementation type);
#endif
