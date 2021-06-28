#include "includes.h"


__global__ void add(long* a, long* b, long* c, long N) { //core from ScalarMultiplication_example1
long baseIdx = threadIdx.x;
long idx = baseIdx;
while (idx < N)
{
c[idx] = a[idx] * b[idx];
idx += blockDim.x;
}
__syncthreads();
long step = N / 2;
while (step != 0) {
idx = baseIdx;
while (idx < step) {
c[idx] += c[idx + step];
idx += blockDim.x;
}
step /= 2;
__syncthreads();
}
}