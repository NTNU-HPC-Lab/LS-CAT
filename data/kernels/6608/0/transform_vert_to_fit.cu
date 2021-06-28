#include "includes.h"
__global__ static void transform_vert_to_fit(const int* src, int* dst, const int nb_vert)
{
const int p = blockIdx.x * blockDim.x + threadIdx.x;
if(p < nb_vert) dst[p] = src[p] < 0 ? 0 : 1;
}