#include "includes.h"
__global__ void prime_generator(int *input,int *prime_list,int *total_input,int *seed)
{

printf("-------XXXXXX>>> %d\n",seed[0]);
int i= blockIdx.x * blockDim.x + threadIdx.x;
int primeno= prime_list[i];
int total=seed[0]*seed[0];
for(int k=seed[0];k<total;k++)
{
if(k%primeno==0)
{
input[k]=1;


}


}



}