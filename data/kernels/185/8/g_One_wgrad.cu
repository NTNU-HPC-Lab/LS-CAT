#include "includes.h"
__global__ void g_One_wgrad( float* _inputs, float* _curDelta, float* _wgradTmp, int rows, int cols, int channels)
{
int row     = blockIdx.x;
int channel = blockIdx.y;

int skip = channel * rows * cols + row * cols;
float* inputs   = _inputs   + skip;
float* curDelta = _curDelta + skip;
float* wgradTmp = _wgradTmp + skip;

for(int i = 0; i < cols; i += blockDim.x){
int id = i + threadIdx.x;
if(id < cols){
wgradTmp[id] = inputs[id] * curDelta[id];
}
}
}