#include "includes.h"
__global__ void transposeNaive(float *odata, const float *idata,int idata_rows,int idata_cols)
{

int x = blockIdx.x * TILE_SIZE + threadIdx.x;
int y = blockIdx.y * TILE_SIZE + threadIdx.y;
//int width = gridDim.x * TILE_SIZE;

if(y<idata_rows && x<idata_cols)
odata[x*idata_rows+y] = idata[y*idata_cols+x];
}