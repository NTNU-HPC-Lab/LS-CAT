#include "includes.h"
__global__ void inc(int *array, size_t n){
size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
while (idx < n){
array[idx]++;
idx += blockDim.x*gridDim.x; // grid-stride loop
}
}