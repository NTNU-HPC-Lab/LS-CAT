#include "includes.h"
__global__ void myset(unsigned long long *p, unsigned long long v, long long n) {
const long long tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < n) {
p[tid] = v;
}
return;
}