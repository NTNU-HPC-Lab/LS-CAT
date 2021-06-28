#include "includes.h"
__global__ void delta_product ( const float * w_ik, const float * d_k, float * output, unsigned int width )
{
// X is layer[i] nodes (size_i)
int x = blockIdx.x * blockDim.x + threadIdx.x;
// Y is layer[k] nodes (size_k) == d_k == w_per_n
int y = blockIdx.y * blockDim.y + threadIdx.y;
//  W[ik] * δ[k] - Row-Major Matrix
output[width*x+y] = __fmul_rz( d_k[y], w_ik[width*x+y]);
}