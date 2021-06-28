#include "includes.h"
__global__ void gradient_descent ( float * d_k, float * o_i, float * g_ik, unsigned int size_d )
{
// X = Node Delta Count (layer k)
int x = blockIdx.x * blockDim.x + threadIdx.x;
// Y = Node Output Count (layer i)
int y = blockIdx.y * blockDim.y + threadIdx.y;
// Row-Major Matrix
g_ik[size_d*x+y] = __fmul_rz( d_k[x], o_i[y]);
}