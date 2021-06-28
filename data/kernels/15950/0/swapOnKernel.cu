#include "includes.h"




cudaError_t sortWithCuda(int *a, size_t size, float* time);

typedef long long int64;
typedef unsigned long long uint64;
__global__ void swapOnKernel(int *a, int size)
{
int i = blockDim.x * blockIdx.x + threadIdx.x * 2;
int cacheFirst;
int cacheSecond;
int cacheThird;

for (int j = 0; j < size/2 + 1; j++) {

if(i+1 < size) {
cacheFirst = a[i];
cacheSecond = a[i+1];

if(cacheFirst > cacheSecond) {
int temp = cacheFirst;
a[i] = cacheSecond;
cacheSecond = a[i+1] = temp;
}
}

if(i+2 < size) {
cacheThird = a[i+2];
if(cacheSecond > cacheThird) {
int temp = cacheSecond;
a[i+1] = cacheThird;
a[i+2] = temp;
}
}

__syncthreads();
}

}