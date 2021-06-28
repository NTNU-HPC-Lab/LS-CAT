#include "includes.h"
__global__ void GPU_simple_power_and_interbin_kernel(float2 *d_input_complex, float *d_output_power, int nTimesamples, float norm){
int pos_x = blockIdx.x*blockDim.x + threadIdx.x;
int pos_y = blockIdx.y*nTimesamples;

float2 A;
A.x = 0; A.y = 0;

if( pos_x < nTimesamples ) {
A = d_input_complex[pos_y + pos_x];
d_output_power[pos_y + pos_x] = (A.x*A.x + A.y*A.y)*norm;
}
}