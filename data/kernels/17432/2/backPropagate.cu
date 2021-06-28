#include "includes.h"

#define BLOCK_SIZE 32


__device__ float deriv_error(float d_output, float d_actual, float d_weights )
{
//float de_dout = d_output - d_actual; //previous derivative error
//float dout_dnet = d_output[] * (1-output[i]);
//de_dout * dout_dnet; // * sum(weights);
return 1.0f;
}
__global__ void backPropagate(float *deriv_err, float *prev_deriv_err, float *wieghts, float *output)
{
//use map operation to multiply d_output[i]*(1-output[i])*prev_deriv_error[i]*weight[i]
//use gather operation to gather all these together.
}