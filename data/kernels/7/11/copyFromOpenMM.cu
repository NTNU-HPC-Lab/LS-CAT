#include "includes.h"
__global__ void copyFromOpenMM( float *target, float *source, int N ) {
const int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
if( elementNum > N ) {
return;
}

const int atom = elementNum / 3;
target[elementNum] = source[4 * atom + elementNum % 3];
}