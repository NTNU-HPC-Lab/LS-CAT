#include "includes.h"

#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))

//data generator
__device__ uint bfe(uint x, uint start, uint nbits)
{
uint bits;
asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
return bits;
}
__global__ void histogram(int* Hist, long long* arrayofkeys, long long size,int noofpartitions)
{
register int thd = threadIdx.x;
register int bD = blockDim.x;
register int bI = blockIdx.x;
uint h,start,nbits;

long long thdindex= bD * bI + thd;
extern __shared__ int sharedpartitions[];
int * sharedHist = (int *)&sharedpartitions[noofpartitions];

for(int m =thd;m<noofpartitions;m=m+bD)
sharedHist[m]=0;

__syncthreads();

start=0;
nbits=(uint)ceil(log2((float)noofpartitions));
if(thdindex<size)
{

h=bfe(arrayofkeys[thdindex],start,nbits);
atomicAdd(&(sharedHist[h]),1);
}
__syncthreads();

for(int n=thd;n<noofpartitions;n=n+bD)
atomicAdd(&(Hist[n]),(sharedHist[n]));

}