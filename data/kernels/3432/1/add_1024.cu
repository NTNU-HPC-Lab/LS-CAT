#include "includes.h"


__global__ void add_1024(long* a, long* b, long* c, long N) { //more simple and probably faster core but works only with 1024 or less elements in vector in this example
c[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
__syncthreads();
long step = N / 2;
while (step != 0) {
if (threadIdx.x < step)
{
c[threadIdx.x] += c[threadIdx.x + step];
}
step /= 2;
__syncthreads();
}
}