#include "includes.h"
__global__ void addValue(int *array_val, int *b_array_val) {
int cacheIndex = threadIdx.x;
int i = blockDim.x/2;
while (i > 0) {
if (cacheIndex < i) {
array_val[blockIdx.x * COLUMNS +cacheIndex] += array_val[blockIdx.x * COLUMNS + cacheIndex +i];
}
__syncthreads();
i /=2;
}
if (cacheIndex == 0)
b_array_val[blockIdx.x] = array_val[blockIdx.x * COLUMNS];

}