#include "includes.h"
__global__ void kernel_forwardElimination( float * fullMatrix, float * B, unsigned int nComp ) {
unsigned int t = threadIdx.x;
unsigned int baseIndex = t*nComp*nComp;

unsigned int i,j,k;
for ( i = 0; i < nComp - 1; i++ )
for ( j = i + 1; j < nComp; j++ ) {
double div = fullMatrix[baseIndex+ j*nComp+i ] / fullMatrix[baseIndex+ i*nComp+ i ];
for ( k = 0; k < nComp; k++ )
fullMatrix[ baseIndex+j*nComp+k ] -= div * fullMatrix[baseIndex+ i *nComp+ k ];
B[ baseIndex+j ] -= div * B[ baseIndex+i ];
}
__syncthreads();
}