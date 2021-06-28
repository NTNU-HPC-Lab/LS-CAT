#include "includes.h"
__global__ void copyToOpenMM( float *target, float *source, int N ) {
int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
int atom = elementNum / 3;
if( elementNum > N ) {
return;
}
//else target[elementNum] = source[elementNum];
else {
target[4 * atom + elementNum % 3] = source[elementNum];
}
}