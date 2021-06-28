#include "includes.h"
__global__ void swan_fast_fill( uint4 *ptr, int len ) {
int idx = threadIdx.x + blockDim.x * blockIdx.x;
if( idx<len) {
ptr[idx] = make_uint4( 0,0,0,0 );
}
}