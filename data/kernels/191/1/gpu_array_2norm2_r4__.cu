#include "includes.h"
__global__ void gpu_array_2norm2_r4__(size_t arr_size, const float *arr, float *bnorm2)
/** Computes the squared Euclidean (Frobenius) norm of an array arr(0:arr_size-1)
INPUT:
# arr_size - size of the array;
# arr(0:arr_size-1) - array;
OUTPUT:
# bnorm2[0:gridDim.x-1] - squared 2-norm of a sub-array computed by each CUDA thread block;
**/
{
size_t i,n;
float _thread_norm2;
extern __shared__ float thread_norms2_r4[];

n=gridDim.x*blockDim.x; _thread_norm2=0.0f;
for(i=blockIdx.x*blockDim.x+threadIdx.x;i<arr_size;i+=n){_thread_norm2+=arr[i]*arr[i];}
thread_norms2_r4[threadIdx.x]=_thread_norm2;
__syncthreads();
if(threadIdx.x == 0){
bnorm2[blockIdx.x]=thread_norms2_r4[0]; for(i=1;i<blockDim.x;i++){bnorm2[blockIdx.x]+=thread_norms2_r4[i];}
}
__syncthreads();
return;
}