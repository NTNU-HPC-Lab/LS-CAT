#include "includes.h"
__global__ void symmetrize1D( float *h, int *blockPositions, int *blockSizes, int numBlocks ) {
int blockNum = blockIdx.x * blockDim.x + threadIdx.x;
if( blockNum >= numBlocks ) {
return;
}

// blockSizes are given in terms of atoms, convert to dof
const unsigned int blockSize = 3 * blockSizes[blockNum];

float *block = &( h[blockPositions[blockNum]] );
for( unsigned int r = 0; r < blockSize - 1; r++ ) {
for( unsigned int c = r + 1; c < blockSize; c++ ) {
const float avg = 0.5f * ( block[r * blockSize + c] + block[c * blockSize + r] );
block[r * blockSize + c] = avg;
block[c * blockSize +	r] = avg;
}
}
}