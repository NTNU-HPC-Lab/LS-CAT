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
__global__ void initpopulation(chromosome *cudaChromo,int seed,const int numofeles,int *devValue,int* devWeight)
{
if (blockIdx.x < N){
int idx = (threadIdx.x + blockIdx.x*blockDim.x);
curandState state;
curand_init(seed, idx, 1, &state);
idx %= numofeles;
bool tmp = curand(&state) % 2 == 1 ? true : false;
cudaChromo[blockIdx.x].chromo[idx] = tmp;
}
}