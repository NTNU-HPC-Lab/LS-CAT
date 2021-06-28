#include "includes.h"
__global__ void RadixHistogram_device( int *dptrHistogram, const int *in, size_t N, int shift, int mask )
{
for ( int i = blockIdx.x*blockDim.x+threadIdx.x;
i < N;
i += blockDim.x*gridDim.x ) {
int index = (in[i] & mask) >> shift;
atomicAdd( dptrHistogram+index, 1 );
}
#if 0
const int cBuckets = 1<<b;
__shared__ unsigned char sharedHistogram[NUM_THREADS][cBuckets];

for ( int i = blockIdx.x*blockDim.x+threadIdx.x;
i < N;
i += blockDim.x*gridDim.x ) {
int index = (in[i] & mask) >> shift;
if ( 0 == ++sharedHistogram[threadIdx.x][index] ) {
atomicAdd( dptrHistogram+index, 256 );
}
}
__syncthreads();
for ( int i = 0; i < cBuckets; i++ ) {
if ( sharedHistogram[threadIdx.x][i] ) {
atomicAdd( dptrHistogram+i, sharedHistogram[threadIdx.x][i] );
}
}
#endif
}