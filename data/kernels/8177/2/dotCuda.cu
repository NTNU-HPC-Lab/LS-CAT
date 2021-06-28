#include "includes.h"

using namespace std;


// https://stackoverflow.com/questions/26853363/dot-product-for-dummies-with-cuda-c



__global__ void dotCuda(float* tmp, float* t1, float* t2, int size) {
//unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

tmp[i] = t1[i] * t2[i];
__syncthreads();

int mididx = size / 2;

while (i < mididx) {
tmp[i] += tmp[i + mididx];
mididx /= 2;
__syncthreads();
}
//atomicAdd(tmp, p);
}