#include "includes.h"

#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))

//data generator
__device__ uint bfe(uint x, uint start, uint nbits)
{
uint bits;
asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
return bits;
}
__global__ void Reorder(long long* arrayofkeys, int* Hist_pre, int noofpartitions, long long size, long long* output)
{
register int thd = threadIdx.x;
register int bD = blockDim.x;
register int bI = blockIdx.x;
uint h,start,nbits;

start=0;
nbits=(uint)ceil(log2((float)noofpartitions));

long long thdindex= bD * bI + thd;

if(thdindex<size)
{
h=bfe(arrayofkeys[thdindex],start,nbits);
int offset=atomicAdd(&(Hist_pre[h]),1);
output[offset] = arrayofkeys[thdindex];

}

}