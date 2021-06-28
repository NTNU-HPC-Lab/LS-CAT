#include "includes.h"
__global__  void elmsk(float *inA, float *inB, bool  *msk, int length)
{
int idx = threadIdx.x + blockDim.x*blockIdx.x;

if (idx<length) {
if (msk[idx]>0) inA[idx] *= inB[idx];
else  inA[idx] = 0;
}
}