#include "includes.h"
__global__ void indices(){
int id=threadIdx.x + blockIdx.x*blockDim.x;
printf("blockdimy: %d  threadx: %d  Blockidx: %d  blockdimx: %d id:  %d raiz: %f\n",
blockDim.y,threadIdx.x , blockIdx.x,blockDim.x, id,sqrt((double)id));
__syncthreads();

}