#include "includes.h"
__global__ void debug_ker(float* ptr, int addr){
//int i = blockIdx.x*blockDim.x + threadIdx.x;
printf("%d %f\n", addr, ptr[addr]);
}