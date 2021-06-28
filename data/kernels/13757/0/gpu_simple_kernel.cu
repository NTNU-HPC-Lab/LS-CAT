#include "includes.h"
__global__ void gpu_simple_kernel(float* a, float* b, float* c, int N)
{
//int thread_idx = threadIdx.x;

int idx = blockIdx.x*blockDim.x + threadIdx.x;
if ( idx > N)
return;

#define PRINT_IDS
#if !defined( __CUDA_ARCH__) || (__CUDA_ARCH__ >= 200 ) &&  defined(PRINT_IDS)
// Check nvcc compiler gencode
// at least -gencode=arch=compute_20,code=\"sm_20,compute_20\" should be set
printf("thread: %3d - block: %3d - threadIdx: %3d, warp: %3d\n", idx, blockIdx.x, threadIdx.x, threadIdx.x/warpSize );
#endif

c[idx] = a[idx] * b[idx];
}