#include "includes.h"
/**
* Various matrix utils using cuda
**/


/**
* Kronecker product of two matrices kernel
* input :
* a : first matrix
* nax, nay : matrix a dimensions
* b: second matrix
* nbx, nby : matrix b dimensions
* results : kronecker product of a and b
**/

__global__ void reduceSum(double * d_arr, const size_t sz, double * d_out)
{
extern __shared__ double sh_out [];
int myId = threadIdx.x + blockDim.x * blockIdx.x;
int tId = threadIdx.x;
if ( myId >= sz)
{
sh_out[tId] = 0.0;
}
else
{
// Fill in the shared memory
sh_out[tId] = d_arr[myId];
}
__syncthreads();
for  (unsigned int s = blockDim.x /2; s > 0; s >>=1)
{
if (tId < s)
{
sh_out[tId] += sh_out[tId+s];
}
__syncthreads();
}
if (tId == 0)
{
d_out[blockIdx.x] = sh_out[0];
}
}