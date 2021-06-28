#include "includes.h"
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif
float a[1024][1024], b[1024][1024], c[1024][1024];




// Now launch your kernel using the appropriate macro:

// Now launch your kernel using the appropriate macro:
//kernel KERNEL_ARGS2(dim3(nBlockCount), dim3(nThreadCount)) (param1);



//matrix multiplication on GPU
__global__ void MMul(float*m, float*d, float*p, int n) {
int r = blockIdx.y*blockDim.y + threadIdx.y;// row
int c = blockIdx.x*blockDim.x + threadIdx.x;//column
float p_sum = 0;

for (int i = 0; i < n; i++) {
p_sum = +m[r*n + i] * d[i*n + c];
}
p[r*n + c] = p_sum;
}