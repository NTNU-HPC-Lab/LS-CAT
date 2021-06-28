#include "includes.h"
__global__ void k_copy_reshape_rowmajor(unsigned int numEls, unsigned int a_nd, const float * a_data, const int * a_dim, const int * a_str, unsigned int z_nd, float * z_data, const int * z_dim, const int * z_str)
{
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < numEls; i += numThreads)
{
const float * a_i = a_data;
int a_ii = i;
for (unsigned int _d = 0; _d < a_nd; ++_d) //make the rightmost coords change fastest
{
unsigned int d = a_nd - _d-1;
int a_i_d = a_ii % a_dim[d];
a_ii = a_ii / a_dim[d];
a_i += a_i_d * a_str[d];
}
int z_ii = i;
float * z_i = z_data;
for (unsigned int _d = 0; _d < z_nd; ++_d) //make the rightmost coords change fastest
{
unsigned int d = z_nd - _d-1;
//i tried to make the for loop count down, but it didn't work!?
int z_i_d = z_ii % z_dim[d];
z_i += z_i_d * z_str[d];
z_ii = z_ii / z_dim[d];
}
z_i[0] = a_i[0]; //copy one lousy float!
}
}