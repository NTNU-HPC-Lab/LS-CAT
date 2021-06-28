#include "includes.h"
__global__ void multVector(int *d1_in, int *d2_in, int *d_out, int n, int m){
int ind = blockDim.x*blockIdx.x + threadIdx.x;
if(ind<m){
d_out[ind]=0;
for(int i=0;i<n;i++){
d_out[ind]+= d1_in[i]*d2_in[i*m+ind];
}
}
}