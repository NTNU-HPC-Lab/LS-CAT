#include "includes.h"
__global__ void g_One_feedforward( float* _inputs, float* _w, float* _b, float* _outputs, int rows, int cols, int channels)
{
int row     = blockIdx.x;
int channel = blockIdx.y;

int skip = channel * rows * cols + row * cols;
float* inputs = _inputs + skip;
float* outputs= _outputs+ skip;
// 	if(threadIdx.x == 0)
// 		sprintf(logStr, "block(%d %d) skip = %d\n", blockIdx.x, blockIdx.y, skip);
float* w = _w + channel * cols;
float* b = _b + channel * cols;

for(int i = 0; i < cols; i += blockDim.x){
int id = i + threadIdx.x;
if(id < cols){
outputs[id] = inputs[id] * w[id] + b[id];
}
}
}