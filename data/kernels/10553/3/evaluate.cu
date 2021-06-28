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
__global__ void evaluate(chromosome *cudaChromo,int *devValue,int *devWeight, int numele)
{
int idx = threadIdx.x+blockDim.x*blockIdx.x;
for (int i = 0; i < numele; i++){
if (cudaChromo[idx].chromo[i])
cudaChromo[idx].value += devValue[i];
cudaChromo[idx].weight += (cudaChromo[idx].chromo[i] ? 1 : 0)*devWeight[i];
}

}