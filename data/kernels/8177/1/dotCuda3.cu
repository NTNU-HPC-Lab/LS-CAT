#include "includes.h"

using namespace std;


// https://stackoverflow.com/questions/26853363/dot-product-for-dummies-with-cuda-c



__global__ void dotCuda3(float *a, float *b, float *c){
__shared__ float cache[1024];
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int cacheIndex = threadIdx.x;
float temp = a[tid] * b[tid];//+ a[tid + blockDim.x] * b[tid + blockDim.x];
cache[cacheIndex] = temp;
__syncthreads();

for (unsigned int i = blockDim.x >> 1; i > 0; i >>= 1) {
if (cacheIndex < i)
cache[cacheIndex] += cache[cacheIndex + i];
__syncthreads();
}

if (cacheIndex == 0){
c[blockIdx.x] = cache[0];
}
}