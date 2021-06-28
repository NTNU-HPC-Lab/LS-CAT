#include "includes.h"
__global__ void gpu_get_neighors(int *neighbors, int n , int k)
{
for (int off1 = 0; off1 < n/gridDim.x+1 ; off1++)
{
for(int off2 = 0; off2 < n/blockDim.x+1 ;off2++){

int m = blockIdx.x+off1*gridDim.x;
int l = threadIdx.x+off2*blockDim.x;

int counter_i =0;
if(m<n && l<n){
for (int i = m-(k/2); i <= m+(k/2); i++)
{
int counter_j=0;
for (int j = l-(k/2); j <= l+(k/2); j++)
{
int index , index_i , index_j;
index = m*n*k*k + l*k*k + counter_i*k +counter_j;
index_i =(n+i)%n;
index_j=(n+j)%n;
neighbors[index] = index_i*n+index_j;
counter_j++;
}
counter_i++;
}
}
}
}
}