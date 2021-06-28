#include "includes.h"
__global__ void global_memory_kernel(int *d_go_to_state, unsigned int *d_failure_state, unsigned int *d_output_state, unsigned char *d_text, unsigned int *d_out, size_t pitch, int m, int n, int p_size, int alphabet, int num_blocks ) {

int idx = blockIdx.x * blockDim.x + threadIdx.x;
int effective_pitch = pitch / sizeof ( int );

int chars_per_block = n / num_blocks;

int start_block = blockIdx.x * chars_per_block;
int stop_block = start_block + chars_per_block;

int chars_per_thread = ( stop_block - start_block ) / blockDim.x;

int start_thread = start_block + chars_per_thread * threadIdx.x;
int stop_thread;
if( blockIdx.x == num_blocks -1 && threadIdx.x==blockDim.x-1)
stop_thread = n - 1;
else stop_thread = start_thread + chars_per_thread + m-1;

int r = 0, s;

int column;

for ( column = start_thread; ( column < stop_thread && column < n ); column++ ) {

while ( ( s = d_go_to_state[r * effective_pitch + (d_text[column]-(unsigned char)'A')] ) == -1 )
r = d_failure_state[r];
r = s;

d_out[idx] += d_output_state[r];
}
}