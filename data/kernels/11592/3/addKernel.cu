#include "includes.h"
__global__ void addKernel(int *c, const int *a)
{
int i = threadIdx.x;
extern __shared__ int smem[];
smem[i] = a[i];
__syncthreads();


if(i == 0)  // 0号线程做平方和
{
c[0] = 0;
for(int d = 0; d < 5; d++)
{
c[0] += smem[d] * smem[d];

}
}
if(i == 1)//1号线程做累加
{
c[1] = 0;
for(int d = 0; d < 5; d++)
{
c[1] += smem[d];
}
}
if(i == 2)  //2号线程做累乘
{
c[2] = 1;
for(int d = 0; d < 5; d++)
{
c[2] *= smem[d];
}
}
}