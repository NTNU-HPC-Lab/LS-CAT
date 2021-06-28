#include "includes.h"
__global__ void cuda_hello() {

printf("Hello from GPU blockIdx.x=%d blockIdx.y=%d blockIdx.z=%d threadIdx.x=%d threadIdx.y=%d threadIdx.z=%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}