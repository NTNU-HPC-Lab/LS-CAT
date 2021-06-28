#include "includes.h"
__global__ void g_One_backpropagation( float* _curDelta, float* _w, float* _nextDelta, int rows, int cols, int channels)
{
int row     = blockIdx.x;
int channel = blockIdx.y;

int skip = channel * rows * cols + row * cols;
float* curDelta = _curDelta + skip;
float* nextDelta= _nextDelta+ skip;

float* w = _w + channel * cols;

for(int i = 0; i < cols; i += blockDim.x){
int id = i + threadIdx.x;
if(id < cols){
nextDelta[id] = curDelta[id] * w[id];
}
}
}