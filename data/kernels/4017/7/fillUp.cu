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

__global__ void fillUp(size_t * d_dst, size_t N){
int myId = blockIdx.x * blockDim.x + threadIdx.x;
if (myId >= N)
return;
d_dst[myId] = myId;
}