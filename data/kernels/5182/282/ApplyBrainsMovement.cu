#include "includes.h"
__global__ void ApplyBrainsMovement( float *CCXY, int dim_XY, float *movement, int dim_movement, int max_clusters ){
int id = blockDim.x*blockIdx.y*gridDim.x   + blockDim.x*blockIdx.x   + threadIdx.x;
if (id<max_clusters){
//--- move in XY
if (dim_movement>=2){
CCXY[id*dim_XY]   -= movement[0];
CCXY[id*dim_XY+1] -= movement[1];
}
//--- apply rotation in X
if (dim_movement>=3){
}
}
}