#include "includes.h"
__global__ void bin(unsigned short *d_input, float *d_output, int in_nsamp) {

int c = ( ( blockIdx.y * BINDIVINF ) + threadIdx.y );
int out_nsamp = ( in_nsamp ) / 2;
int t_out = ( ( blockIdx.x * BINDIVINT ) + threadIdx.x );
int t_in = 2 * t_out;

size_t shift_one = ( (size_t)(c*out_nsamp) + (size_t)t_out );
size_t shift_two = ( (size_t)(c*in_nsamp)  + (size_t)t_in );

d_output[( shift_one )] = (float) ( ( d_input[( shift_two )] + d_input[(size_t)(shift_two + 1)] )/2.0f );

}