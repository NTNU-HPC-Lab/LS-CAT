#include "includes.h"
__global__ void backward_sam_kernel(float *in_w_h_c_delta, int size, int channel_size, float *in_scales_c, float *out_from_delta, float *in_from_output, float *out_state_delta)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;
if (index < size) {
out_state_delta[index] += in_w_h_c_delta[index] * in_from_output[index]; // l.delta * from  (should be divided by channel_size?)
out_from_delta[index] += in_scales_c[index] * in_w_h_c_delta[index]; // input * l.delta

//out_state_delta[index] += in_w_h_c_delta[index];
//out_from_delta[index] = in_w_h_c_delta[index];
}
}