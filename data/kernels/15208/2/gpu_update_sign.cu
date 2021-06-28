#include "includes.h"
__global__ void gpu_update_sign(int *G, double *w ,int *neighbors , int k , int n ,int *temp, int *flag,int it_b ,int it_t)
{

int buf=0;

for (int off1 = 0; off1 < it_b; off1++)
{
for(int off2 = 0; off2<it_t;off2++){
int result;
double sum = 0.0;

int x = blockIdx.x+off1*gridDim.x;
int y = threadIdx.x+off2*blockDim.x;

if(x<n && y<n){
for (int i = 0; i < k; i++){
for (int j = 0; j < k; j++){
sum += ((double)G[neighbors[x*n*k*k+y*k*k+i*k+j]])*w[i*k+j];
}
}

if ( sum > 1e-6){
result = 1;
if (result != G[neighbors[x*n*k*k+y*k*k+12]])
buf++;
}
else if( sum < -(1e-6)){
result = -1;
if (result != G[neighbors[x*n*k*k+y*k*k+12]])
buf++;
}
else{
result = G[neighbors[x*n*k*k+y*k*k+12]];
}
temp[x*n+y] =result;
}
}
}
*flag+=buf;
__syncthreads();
}