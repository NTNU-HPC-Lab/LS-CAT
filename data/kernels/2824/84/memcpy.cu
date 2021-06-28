#include "includes.h"
__global__ void memcpy( float *dst, float *src )
{

int index = threadIdx.x + 4 * blockIdx.x * blockDim.x;
float a[4];//allocated in registers
for(int i=0;i<4;i++) a[i]=src[index+i*blockDim.x];
for(int i=0;i<4;i++) dst[index+i*blockDim.x]=a[i];
}