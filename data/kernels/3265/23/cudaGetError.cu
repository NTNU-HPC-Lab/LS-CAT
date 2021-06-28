#include "includes.h"
__global__ void cudaGetError(int N, double *ana, double *cur, double *e_sum){
// Parallelly compute the error
int index = blockIdx.x*blockDim.x + threadIdx.x;
if(index < (N+1)*(N+1)) (*e_sum) += (ana[index] - cur[index])*(ana[index] - cur[index]);
return;
}