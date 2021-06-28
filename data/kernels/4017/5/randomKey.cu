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

__global__ void randomKey(size_t N, float * d_dst, unsigned long seed){
int myId = blockIdx.x * blockDim.x + threadIdx.x;
if (myId >= N)
return;
curandState state;
curand_init ( seed, myId, 0, &state);
float RANDOM = curand_uniform( &state );
d_dst[myId] = (float)RANDOM;
}