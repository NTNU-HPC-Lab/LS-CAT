#include "includes.h"
__global__ void update_population_metadata( unsigned int * pop , unsigned int rows , unsigned int cols , unsigned int * free , unsigned int * lost , unsigned int * fixed ) {
unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

__shared__ unsigned int sPop[ MAX_THREADS ];
__shared__ unsigned int sMeta[ MAX_THREADS ];

bool is_fixed_thread = ((threadIdx.y & 1) == 0);
unsigned int eoffset = tid + ((is_fixed_thread) ? blockDim.x : -blockDim.x);
sMeta[ tid ] = ((is_fixed_thread) ? -1 : 0);
__syncthreads();

unsigned int b_offset = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int offset = threadIdx.y * cols + b_offset;
unsigned int mat_size = rows * cols;

unsigned int i = 0;

// scan population
while( i < rows ) {
sPop[ tid ] = ((offset < mat_size) ? pop[offset] : 0 );
__syncthreads();

unsigned int d = sPop[tid], e = sPop[eoffset];
__syncthreads();

unsigned int res = sMeta[ tid ];
if( offset < mat_size ) {
// should only fail when rows is NOT a multiple of THREAD_ROWS
if( is_fixed_thread ) {
res &= (d & e);
} else {
res |= (d | e);
}
}
__syncthreads();

sMeta[ tid ] = res;
__syncthreads();

i += blockDim.y;
offset += blockDim.y * cols;
}

// reduce the fixed and lost lists
i = 4;
while( i <= 32 ) {
unsigned int masked = (threadIdx.y & (i - 1));
unsigned int t =  ((tid + (i / 2) * blockDim.x) & (MAX_THREADS - 1));

// how will branches execute?
// assuming that threads are grouped into warps according to their threadIdx.x coordinate
// all threads in a warp should execute same logic
//
unsigned int res = sMeta[tid], v = sMeta[t];
__syncthreads();

if( masked == 0 ) {
res &= v;
} else if( masked == 1 ) {
res |= v;
}
__syncthreads();

sMeta[ tid ] = res;
__syncthreads();
i <<= 1;
}

// use a single warp to write shared data back to global memory
if( threadIdx.y == 0 ) {
unsigned int fxd = sMeta[ threadIdx.x ];
unsigned int lst = (~sMeta[ blockDim.x + threadIdx.x ]);

free[ b_offset ] = (fxd | lst);
fixed[ b_offset ] = fxd;
lost[ b_offset ] = lst;
}
__syncthreads();
}