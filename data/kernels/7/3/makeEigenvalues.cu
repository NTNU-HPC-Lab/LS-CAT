#include "includes.h"
__global__ void makeEigenvalues( float *eigenvalues, float *blockHessian, int *blocknums, int *blocksizes, int *hessiannums, int N, int numblocks ) {
// elementnum is the degree of freedom (0 to 3n-1)
int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
if( elementNum >= N ) {
return;
}

// b is the block number in which DOF elementnum resides
// blocknums contains atom numbers, so we must divide by 3
// We find the first index with an atom number larger than
// ours, and take one less (or numblocks-1 if we are at the end)
int b = 0;
while( b < numblocks ) {
if( blocknums[b] > elementNum / 3 ) {
break;
}
b++;
}
b--;

// 3*blocknums[b] is the starting degree of freedom for our block
// We must compute an offset from that, call it x.
int x = elementNum - 3 * blocknums[b];

// We initialize our spot to hessiannums[b], which is the starting
// Hessian location for our block.
// We then want to take the diagonal entry from that offset
// So element (x,x)
int spot = hessiannums[b] + x * ( 3 * blocksizes[b] ) + x;

eigenvalues[elementNum] = blockHessian[spot];
}