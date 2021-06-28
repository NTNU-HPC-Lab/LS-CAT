#include "includes.h"
__global__ void blockEigSort( float *eigenvalues, float *eigenvectors, int *blocknums, int *blocksizes, int N ) {
int blockNumber = blockIdx.x * blockDim.x + threadIdx.x;
int startspot = blocknums[blockNumber];
int endspot = startspot + blocksizes[blockNumber] - 1;

// Bubble sort for now, thinking blocks are relatively small
// We may fix it later
for( int i = startspot; i < endspot; i++ ) {
for( int j = startspot; j < i; j++ ) {
if( eigenvalues[j] > eigenvalues[j + 1] ) {
float tmp = eigenvalues[j];
eigenvalues[j] = eigenvalues[j + 1];
eigenvalues[j + 1] = tmp;

// Swapping addresses
for( int i = 0; i < N; i++ ) {
tmp = eigenvectors[i * N + j];
eigenvectors[i * N + j] = eigenvectors[i * N + j + 1];
eigenvectors[i * N + j + 1] = tmp;
}
/*float* tmpaddr = eigenvectors[j];
eigenvectors[j] = eigenvectors[j+1];;
eigenvectors[j+1] = tmpaddr;*/
}
}
}
}