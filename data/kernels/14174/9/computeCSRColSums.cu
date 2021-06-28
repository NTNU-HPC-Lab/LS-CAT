#include "includes.h"
__device__ static void myAtomicAdd(float *address, float value)
{
#if __CUDA_ARCH__ >= 200
atomicAdd(address, value);
#else
// cf. https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks
int oldval, newval, readback;

oldval = __float_as_int(*address);
newval = __float_as_int(__int_as_float(oldval) + value);
while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval)
{
oldval = readback;
newval = __float_as_int(__int_as_float(oldval) + value);
}
#endif
}
__global__ void computeCSRColSums(float *d_colSums, const float *d_systemMatrixVals, const int *d_systemMatrixRows, const int *d_systemMatrixCols, const size_t m, const size_t n)
{
const size_t row = blockIdx.x * blockDim.x + threadIdx.x;

if (row >= m)
return;

for (size_t cidx = d_systemMatrixRows[row]; cidx < d_systemMatrixRows[row+1]; ++cidx)
{
myAtomicAdd(d_colSums + d_systemMatrixCols[cidx], d_systemMatrixVals[cidx]);
}
}