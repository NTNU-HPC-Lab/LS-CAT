#include "includes.h"
__global__ void UpdateCC_XY( float *CCXY, int id_CC, float *XY_tofill, int dim_XY ){
int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x	+ threadIdx.x;
if(id < dim_XY)
CCXY[id_CC*dim_XY + id] = XY_tofill[id];
}