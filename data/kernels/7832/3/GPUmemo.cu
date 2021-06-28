#include "includes.h"
__global__ void GPUmemo( float *data, int pts )
{
__shared__ float* trace;

trace = (float *)malloc(pts*sizeof(float));
int Blocks;
for( Blocks = 0; Blocks < gridDim.x; Blocks++ )
{
trace[threadIdx.x] = data[threadIdx.x + Blocks*pts];
}
}