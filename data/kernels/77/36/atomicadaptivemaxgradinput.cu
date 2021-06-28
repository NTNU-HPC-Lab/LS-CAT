#include "includes.h"
__global__ void atomicadaptivemaxgradinput( float *gradInput, float *gradOutput, float *indices_x, float *indices_y, int input_n, int input_h, int input_w, int output_h, int output_w )
{
// iterators
int xx, yy;

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
indices_x = indices_x + o*output_w*output_h;
indices_y = indices_y + o*output_w*output_h;

// compute gradInput
for(yy = yy_start; yy < yy_end; yy+=yy_step) {

int y_start = (int)floor(float(yy) / output_h * input_h);

for(xx = xx_start; xx < xx_end; xx+=xx_step) {

int x_start = (int)floor(float(xx) / output_w * input_w);

float *ptr_gradInput = gradInput + y_start*input_w + x_start;
float *ptr_gradOutput = gradOutput + yy*output_w + xx;
float *ptr_ind_x = indices_x + yy*output_w + xx;
float *ptr_ind_y = indices_y + yy*output_w + xx;
float z = *ptr_gradOutput;

int argmax_x = (*ptr_ind_x)-1;
int argmax_y = (*ptr_ind_y)-1;

// atomic add since different threads could update same variable
atomicAdd(&(ptr_gradInput[argmax_x + argmax_y*input_w]), z);
}
}
}