#include "includes.h"
// tatami.cu


const unsigned nMax(100000000);
const unsigned nMaxSqrt(sqrt(nMax));



__global__ void odd(unsigned* v, unsigned base)
{
unsigned i = (blockIdx.x * blockDim.x + threadIdx.x + base) * 2 + 7;
unsigned k2 = i + 3;
unsigned k3 = i + i - 4;
while ((k2 <= k3) && ((i * k2) < nMax))
{
unsigned k4 = (nMax - 1) / i;
if (k3 < k4)
k4 = k3;
__syncthreads();
for (unsigned j = k2 / 2; j <= k4 / 2; j++)
atomicInc(&v[i * j], 0xffffffff);
__syncthreads();
k2 += i + 1;
k3 += i - 1;
}
}