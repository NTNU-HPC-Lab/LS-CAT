#include "includes.h"
__global__ void gpu_array_init_r4__(size_t tsize, float *arr, float val)
/** arr(:)=val **/
{
size_t _ti = blockIdx.x*blockDim.x + threadIdx.x;
size_t _gd = gridDim.x*blockDim.x;
for(size_t l=_ti;l<tsize;l+=_gd){arr[l]=val;}
return;
}