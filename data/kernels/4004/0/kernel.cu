#include "includes.h"

#define gpu_assert(rv) gpu_assert_h((rv), __FILE__, __LINE__)
__global__ void kernel()
{
printf("Hello Kernel %d\n", blockIdx.x * blockDim.x + threadIdx.x);
}