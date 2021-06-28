#include "includes.h"
#define N 50
#define NewN 100

#define LifeN 500
#define numofthreads 512
int numofeles=0,capacity;

struct chromosome
{
long long weight=0, value=0;
bool chromo[100003];
};
chromosome chromoele[N],*cudaChromo,*cudaNewpopulation,newpopulation[NewN],res,x[2];
int weight[100001],value[100001],*devValue,*devWeight,*devnumeles;
__global__ void initOne(chromosome *cudaChromo, const int numele,int *devValue,int *devWeight)
{
if (blockIdx.x < N){
int idx = threadIdx.x + blockIdx.x*blockDim.x;
idx %= numele;
if (blockIdx.x == idx)
{
cudaChromo[blockIdx.x].chromo[idx] = true;
cudaChromo[blockIdx.x].value = devValue[idx];
cudaChromo[blockIdx.x].weight = devValue[idx];
}
else
cudaChromo[blockIdx.x].chromo[idx] = false;
}
}