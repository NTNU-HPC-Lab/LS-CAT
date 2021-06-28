#include "includes.h"
__global__ void makeBlockHessian( float *h, float *forces1, float *forces2, float *mass, float blockDelta, int *blocks, int *blocksizes, int numblocks, int *hessiannums, int *hessiansizes, int setnum, int N ) {
int blockNum = blockIdx.x * blockDim.x + threadIdx.x;
int dof = 3 * blocks[blockNum] + setnum;
int atom = dof / 3;
if( atom >= N || ( blockNum != numblocks - 1 && atom >= blocks[blockNum + 1] ) ) {
return;    // Out of bounds
}

int start_dof = 3 * blocks[blockNum];
int end_dof;
if( blockNum == numblocks - 1 ) {
end_dof = 3 * N;
} else {
end_dof = 3 * blocks[blockNum + 1];
}

/* I also would like to parallelize this at some point as well */
for( int k = start_dof; k < end_dof; k++ ) {
float blockScale = 1.0 / ( blockDelta * sqrt( mass[atom] * mass[k / 3] ) );
//h[startspot+i] = (forces1[k] - forces2[k]) * blockScale;
h[hessiannums[blockNum] + ( k - start_dof ) * ( 3 * blocksizes[blockNum] ) + ( dof - start_dof )] = ( forces1[k] - forces2[k] ) * blockScale;
}
}