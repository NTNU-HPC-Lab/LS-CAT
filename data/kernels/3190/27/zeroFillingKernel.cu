#include "includes.h"
__global__ void zeroFillingKernel(float* idata, int row, int length, int height)
{
int tidx = blockIdx.x * blockDim.x + threadIdx.x;
int tidy = blockIdx.y * blockDim.y;
if(tidx < length &&  tidy < height)
{
//printf("idata[%d][%d]: = %f\n", (row+tidy), tidx,idata[tidx + (row+tidy) *length]);
idata[tidx + (row+tidy) *length] = 0;
idata[tidx + (row-tidy) *length] = 0;
//printf("idata[%d][%d]: = %f\n", (row+tidy), tidx,idata[tidx + (row+tidy) *length]);

}
}