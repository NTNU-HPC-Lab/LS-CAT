#include "includes.h"
__global__ void dyadicAdd(int * counter, const int length, const int shift)
{
if (shift > 0) {
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
int adds = 2*shift;
int Index = adds*(xIndex+1)-1;

if (Index < length) {
counter[Index] = counter[Index] + counter[Index-shift];
}
}
}