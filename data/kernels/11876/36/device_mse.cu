#include "includes.h"
__global__ void device_mse ()
{
__shared__ double partial_mse[REDUC_THREADS] ;
int i, index ;
unsigned int n ;
double diff, sum_mse ;

index = threadIdx.x ;
n = d_ncases * d_ntarg ;

sum_mse = 0.0 ;
for (i=blockIdx.x*blockDim.x+index ; i<n ; i+=blockDim.x*gridDim.x) {
diff = d_output[i] - d_targets[i] ;
sum_mse += diff * diff ;
}

partial_mse[index] = sum_mse ;
__syncthreads() ;

for (i=blockDim.x>>1 ; i ; i>>=1) {
if (index < i)
partial_mse[index] += partial_mse[index+i] ;
__syncthreads() ;
}

if (index == 0)
d_mse_out[blockIdx.x] = partial_mse[0] ;
}