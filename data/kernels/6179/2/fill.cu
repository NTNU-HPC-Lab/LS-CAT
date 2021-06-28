#include "includes.h"


__global__ void fill( float4 *localbuf, float val, float4* ptr, int offset, int N ) {
int idx= blockDim.x * blockIdx.x + threadIdx.x;
if( idx < N ) {
float4 t = localbuf[ idx ];
t.x += val;
t.y += val;
t.z += val;
t.w += val;

ptr[ offset + idx ] = t;
}
}