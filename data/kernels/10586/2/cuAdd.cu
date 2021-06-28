#include "includes.h"
#define PI 3.141592653589793
#define BLOCKSIZE 1024


__global__ void cuAdd(float *dst, float *src, int size)
{
int id=blockIdx.x*blockDim.x+threadIdx.x;
if(id>=size) return;
dst[id]+=src[id];
}