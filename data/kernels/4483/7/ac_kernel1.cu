#include "includes.h"
__global__ void ac_kernel1 ( int *d_state_transition, unsigned int *d_state_supply, unsigned int *d_state_final, unsigned char *d_text, unsigned int *d_out, size_t pitch, int m, int n, int p_size, int alphabet, int numBlocks ) {

//int idx = blockIdx.x * blockDim.x + threadIdx.x;
int effective_pitch = pitch / sizeof ( int );

int charactersPerBlock = n / numBlocks;

int startBlock = blockIdx.x * charactersPerBlock;
int stopBlock = startBlock + charactersPerBlock;

int charactersPerThread = ( stopBlock - startBlock ) / blockDim.x;

int startThread = startBlock + charactersPerThread * threadIdx.x;
int stopThread;
if( blockIdx.x == numBlocks -1 && threadIdx.x==blockDim.x-1)
stopThread = n - 1;
else stopThread = startThread + charactersPerThread + m-1;

int r = 0, s;

int column;

//cuPrintf("Working from %i to %i chars %i\n", startThread, stopThread, charactersPerThread);

for ( column = startThread; ( column < stopThread && column < n ); column++ ) {

while ( ( s = d_state_transition[r * effective_pitch + (d_text[column]-(unsigned char)'A')] ) == -1 )
r = d_state_supply[r];
r = s;

d_out[column] = d_state_final[r];
}
}