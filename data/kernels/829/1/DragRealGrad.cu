#include "includes.h"
__global__ void DragRealGrad(float2 *ORIGIN , float *DEST , float *VEC) {
int idx = threadIdx.x + blockIdx.x*blockDim.x;
DEST[idx] = ORIGIN[idx].x/sqV - VEC[idx];
}