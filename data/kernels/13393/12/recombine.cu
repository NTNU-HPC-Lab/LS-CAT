#include "includes.h"
__global__ void recombine( unsigned int * p0 , unsigned int * p1 , unsigned int * off , unsigned int cols ) {
unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

unsigned int boffset = blockIdx.x * blockDim.x + tid;

unsigned int p = ((boffset < cols) ? p0[ boffset ] : 0 );
unsigned int q = ((boffset < cols) ? p1[ boffset ] : 0 );
unsigned int res = ((boffset < cols) ? off[ boffset ] : 0 );
__syncthreads();

res = (( p & ~res ) | ( q & res ));
__syncthreads();

if( boffset < cols ) {
off[ boffset ] = res;
}
}