#include "includes.h"
__global__ void kernel(float* data, size_t from, size_t to, size_t min, size_t max, size_t NX)
{
size_t i = min + blockIdx.x * blockDim.x + threadIdx.x;
while (i < max) {
//TODO CONSIDER REMOVING MODULUS (might be slow)
if ( (i % NX != 0) && (i % NX != NX - 1) ){
data[to+i] = 0.2 * (
data[from+i]
+ data[from+i-1]
+ data[from+i+1]
+ data[from+i-NX]
+ data[from+i+NX]);
}
i +=gridDim.x*blockDim.x;
}
}