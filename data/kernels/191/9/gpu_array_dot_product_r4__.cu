#include "includes.h"
__global__ void gpu_array_dot_product_r4__(size_t tsize, const float *arr1, const float *arr2, volatile float *dprod)
{
extern __shared__ float dprs_r4[]; //volume = blockDim.x
size_t l;
int i,j;
float dpr;
dpr=0.0f; for(l=blockIdx.x*blockDim.x+threadIdx.x;l<tsize;l+=gridDim.x*blockDim.x){dpr+=arr1[l]*arr2[l];}
dprs_r4[threadIdx.x]=dpr;
__syncthreads();
i=1; while(i < blockDim.x){j=threadIdx.x*(i*2); if(j+i < blockDim.x) dprs_r4[j]+=dprs_r4[j+i]; i*=2;}
__syncthreads();
if(threadIdx.x == 0){
i=1; while(i == 1){i=atomicMax(&dot_product_wr_lock,1);} //waiting for a lock to unlock, then lock
*dprod+=dprs_r4[0];
__threadfence();
i=atomicExch(&dot_product_wr_lock,0); //unlock
}
__syncthreads();
return;
}