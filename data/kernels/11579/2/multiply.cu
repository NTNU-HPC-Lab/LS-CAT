#include "includes.h"
__global__ void multiply(int *result, int *A, int *B)
{
/* OLD logic
We have a 3 by 3 grid and each block has 3 threads.
So rows = block x id, cols = block y id
So Indices will be C[block X id][block Y id] = A[block X id][threads 0, 1, 2] * B[threads 0, 1, 2][block y id]
*/
//__shared__ int result[_size*_size] ;
/*result[blockIdx.x*blockDim.x +blockIdx.y] += A[blockIdx.x*blockDim.x + threadIdx.x]*B[blockDim.x*threadIdx.x+blockIdx.y];
printf("C[%d] = A[%d]*B[%d] = %d*%d\n",blockIdx.x*blockDim.x +blockIdx.y, blockIdx.x*blockDim.x + threadIdx.x, blockDim.x*threadIdx.x+blockIdx.y,
A[blockIdx.x*blockDim.x + threadIdx.x],B[blockDim.x*threadIdx.x+blockIdx.y]);
Res[blockIdx.x*blockDim.x +blockIdx.y]= result[blockIdx.x*blockDim.x +blockIdx.y];*/

/* NEW logic
I have 3 blocks and 3 threads. Each thread calculates entry for each position compared to the old one having each thread multiplying one value.
So indices will be result[block x id][thread id] = A[block x id][i]* B[i][thread x id]
*/

for(int i=0; i<_size;i++)
{
result[blockIdx.x*blockDim.x +threadIdx.x] += A[blockIdx.x*blockDim.x+i]*B[blockDim.x*i+threadIdx.x];
}
}