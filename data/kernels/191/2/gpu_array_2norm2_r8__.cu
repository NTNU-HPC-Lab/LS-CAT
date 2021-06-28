#include "includes.h"
__global__ void gpu_array_2norm2_r8__(size_t arr_size, const double *arr, double *bnorm2)
/** Computes the squared Euclidean (Frobenius) norm of an array arr(0:arr_size-1)
INPUT:
# arr_size - size of the array;
# arr(0:arr_size-1) - array;
OUTPUT:
# bnorm2[0:gridDim.x-1] - squared 2-norm of a sub-array computed by each CUDA thread block;
**/
{
size_t i,n;
double _thread_norm2;
extern __shared__ double thread_norms2_r8[];

n=gridDim.x*blockDim.x; _thread_norm2=0.0;
for(i=blockIdx.x*blockDim.x+threadIdx.x;i<arr_size;i+=n){_thread_norm2+=arr[i]*arr[i];}
thread_norms2_r8[threadIdx.x]=_thread_norm2;
__syncthreads();
if(threadIdx.x == 0){
bnorm2[blockIdx.x]=thread_norms2_r8[0]; for(i=1;i<blockDim.x;i++){bnorm2[blockIdx.x]+=thread_norms2_r8[i];}
}
__syncthreads();
return;
}