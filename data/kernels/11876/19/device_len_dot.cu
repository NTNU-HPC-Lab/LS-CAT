#include "includes.h"
__global__ void device_len_dot ()
{
__shared__ float partial_len[REDUC_THREADS], partial_dot[REDUC_THREADS] ;
int i, n, index ;
float sum_len, sum_dot ;

index = threadIdx.x ;
n = d_n_inputs_cols * d_nhid ;

sum_len = sum_dot = 0.0f ;
for (i=blockIdx.x*blockDim.x+index ; i<n ; i+=blockDim.x*gridDim.x) {
sum_len += d_w_grad[i] * d_w_grad[i] ;
sum_dot += d_w_grad[i] * d_prev_grad[i] ;
d_prev_grad[i] = d_w_grad[i] ;
}

partial_len[index] = sum_len ;
partial_dot[index] = sum_dot ;
__syncthreads() ;

for (i=blockDim.x>>1 ; i ; i>>=1) {
if (index < i) {
partial_len[index] += partial_len[index+i] ;
partial_dot[index] += partial_dot[index+i] ;
}
__syncthreads() ;
}

if (index == 0) {
d_len_out[blockIdx.x] = partial_len[0] ;
d_dot_out[blockIdx.x] = partial_dot[0] ;
}
}