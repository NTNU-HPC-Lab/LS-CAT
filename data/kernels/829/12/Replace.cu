#include "includes.h"
__global__ void Replace(float *WHAT , float *WHERE)
{

int idx = threadIdx.x + blockIdx.x*blockDim.x;
WHERE[idx] = WHAT[idx];

}