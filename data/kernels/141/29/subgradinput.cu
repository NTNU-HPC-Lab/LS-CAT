#include "includes.h"
__global__ void subgradinput(float *gradInput, float *gradOutput, int input_n, int input_h, int input_w, int kH, int kW, int dH, int dW)
{
// iterators
int xx, yy;

// output size
int output_w = (input_w - kW) / dW + 1;
int output_h = (input_h - kH) / dH + 1;

// compute offsets based on thread/block ID
int o = blockIdx.x;
int i = o;

int xx_start = threadIdx.x;
int xx_end = output_w;
int xx_step = blockDim.x;

int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
int yy_end = output_h;
int yy_step = blockDim.y*gridDim.y;

// select input/output plane
gradOutput = gradOutput + o*output_w*output_h;
gradInput = gradInput + i*input_w*input_h;

// compute gradInput
for(yy = yy_start; yy < yy_end; yy+=yy_step) {
for(xx = xx_start; xx < xx_end; xx+=xx_step) {
float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
float *ptr_gradOutput = gradOutput + yy*output_w + xx;
float z = *ptr_gradOutput;
int kx, ky;
for(ky = 0; ky < kH; ky++) {
for(kx = 0; kx < kW; kx++)
ptr_gradInput[kx] += z / float(kW*kH);
ptr_gradInput += input_w;
}
}
}
}