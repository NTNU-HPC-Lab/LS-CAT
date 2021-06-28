#include "includes.h"

using namespace std;


// https://stackoverflow.com/questions/26853363/dot-product-for-dummies-with-cuda-c



__global__ void reduce0(float* g_odata, float* g_idata1, float* g_idata2) {
extern __shared__ float sdata[];
// each thread loads one element from global to shared mem

unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
sdata[tid] = g_idata1[i] * g_idata2[i];
__syncthreads();
// do reduction in shared mem
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
if (tid < s) {
sdata[tid] += sdata[tid + s];
}
__syncthreads();
}
// write result for this block to global mem
if (tid == 0) {
g_odata[blockIdx.x] = sdata[0];
//atomicAdd(g_odata, sdata[0]);
}
}