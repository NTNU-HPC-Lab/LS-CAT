#include "includes.h"
__device__ void swap(float& a, float& b)
{
float temp = a;
a = b;
b = temp;
}
__global__ void batcherBitonicMergesort64(float * d_out, const float * d_in)
{
// you are guaranteed this is called with <<<1, 64, 64*4>>>
extern __shared__ float sdata[];
int tid  = threadIdx.x;
sdata[tid] = d_in[tid];
__syncthreads();

for (int stage = 0; stage <= 5; stage++)
{
for (int substage = stage; substage >= 0; substage--)
{
int distance = 1 << substage; // Distance to value to be compared
int comparison = tid - distance; // Value to be compared
int div = 1 << (stage + 1);
// Skip values that should not be compared
if (comparison < 0 || (comparison / div) != (tid / div)) {
continue;
}
bool up = (comparison / div) % 2 == 1;
if (up) {
if (sdata[tid] > sdata[comparison]) {
swap(sdata[tid], sdata[comparison]);
}
} else {
if (sdata[tid] < sdata[comparison]) {
swap(sdata[tid], sdata[comparison]);
}
}
}
__syncthreads();
}

d_out[tid] = sdata[tid];
}