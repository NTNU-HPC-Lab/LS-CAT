#include "includes.h"
__global__ void reposition (double4 *ac, double4 *ac1, double4 *ac2, double4 *af, unsigned long nextsize)
{
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

if(i < nextsize){
af[i]              = ac[i];
af[i + nextsize]   = ac1[i];
af[i + 2*nextsize] = ac2[i];
}


}