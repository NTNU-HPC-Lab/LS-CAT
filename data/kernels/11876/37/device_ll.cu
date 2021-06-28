#include "includes.h"
__global__ void device_ll ()
{
__shared__ double partial_ll[REDUC_THREADS] ;
int i, n, ntarg, index ;
double sum_ll ;

index = threadIdx.x ;
n = d_ncases ;
ntarg = d_ntarg ;

sum_ll = 0.0 ;
for (i=blockIdx.x*blockDim.x+index ; i<n ; i+=blockDim.x*gridDim.x)
sum_ll -= log ( d_output[i*ntarg+d_class[i]] + 1.e-30 ) ;

partial_ll[index] = sum_ll ;
__syncthreads() ;

for (i=blockDim.x>>1 ; i ; i>>=1) {
if (index < i)
partial_ll[index] += partial_ll[index+i] ;
__syncthreads() ;
}

if (index == 0)
d_mse_out[blockIdx.x] = partial_ll[0] ;
}