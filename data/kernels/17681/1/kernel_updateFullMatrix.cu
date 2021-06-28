#include "includes.h"
__global__ void kernel_updateFullMatrix( float * device_fullMatrix, float * B, float * V, float * Cm, float * Em, float * Rm, float dt, unsigned int nComp ) {
//TODO: fix memory usage matter

unsigned int t = threadIdx.x;
unsigned int baseIndex = t*nComp;

unsigned int i;
for ( i = 0; i < nComp; i++ )
{
unsigned int myIndex=baseIndex+i;
B[myIndex  ] =
V[ myIndex] * Cm[myIndex] 	/ ( dt / 2.0 ) +
Em[ myIndex] / Rm[myIndex];
}
__syncthreads();
}