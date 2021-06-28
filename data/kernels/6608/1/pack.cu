#include "includes.h"
__global__ static void pack(const int* prefix_sum, const int* src, int* dst, const int nb_vert)
{
const int p = blockIdx.x * blockDim.x + threadIdx.x;
if(p < nb_vert){
const int elt = src[p];
if(elt >= 0) dst[ prefix_sum[p] ] = elt;
}
}