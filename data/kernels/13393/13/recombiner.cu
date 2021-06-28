#include "includes.h"
__global__ void recombiner( double * rands , unsigned int * parents , unsigned int parent_rows , unsigned int parent_cols , unsigned int * off , unsigned int cols , unsigned int seq_offset ) {
double id_offset = rands[ seq_offset + blockIdx.y ];
__syncthreads();

unsigned int col_offset = (blockIdx.x + threadIdx.y) * blockDim.x + threadIdx.x;

// using integer cast to truncate of fractional portion
unsigned int p0_offset = id_offset * ((parent_rows - 1) / 2);
p0_offset = (2 * p0_offset * parent_cols) + col_offset;

unsigned int p = 0, q = 0, res = 0;
if( col_offset < parent_cols ) {
// should hold true for entire warps
p = parents[ p0_offset ];
q = parents[ p0_offset + parent_cols ];
}
__syncthreads();

if( col_offset < cols ) {
res = off[ (seq_offset + blockIdx.y) * cols + col_offset ];
}
__syncthreads();

res = (( p & ~res ) | ( q & res ));
__syncthreads();

if( col_offset < cols ) {
off[ (seq_offset + blockIdx.y) * cols + col_offset ] = res;
}
}