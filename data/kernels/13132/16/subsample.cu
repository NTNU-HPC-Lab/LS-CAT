#include "includes.h"
__global__ void subsample(float *input, float *output, float *weight, float *bias, int input_n, int input_h, int input_w, int kH, int kW, int dH, int dW)
{
// iterators
int xx, yy;

// output size
int output_w = (input_w - kW) / dW + 1;
int output_h = (input_h - kH) / dH + 1;

// compute offsets based on thread/block ID
int o = blockIdx.x;
int i = o;
int k = blockIdx.x % input_n;

int xx_start = threadIdx.x;
int xx_end = output_w;
int xx_step = blockDim.x;

int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
int yy_end = output_h;
int yy_step = blockDim.y*gridDim.y;

// select input/output plane
output = output + o*output_w*output_h;
input = input + i*input_w*input_h;

// Get the good mask for (k,i) (k out, i in)
float the_weight = weight[k];

// Initialize to the bias
float the_bias = bias[k];

// For all output pixels...
for(yy = yy_start; yy < yy_end; yy+=yy_step) {
for(xx = xx_start; xx < xx_end; xx+=xx_step) {
// Compute the mean of the input image...
float *ptr_input = input + yy*dH*input_w + xx*dW;
float *ptr_output = output + yy*output_w + xx;
float sum = 0;
int kx, ky;
for(ky = 0; ky < kH; ky++) {
for(kx = 0; kx < kW; kx++)
sum += ptr_input[kx];
ptr_input += input_w; // next input line
}
// Update output
*ptr_output = the_weight*sum + the_bias;
}
}
}