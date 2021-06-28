#include "includes.h"
__global__ void Frontier_copy( unsigned int *frontier, unsigned int *frontier2, unsigned int *frontier_length)
{
unsigned int tid=threadIdx.x + blockDim.x * blockIdx.x;

if(tid<*frontier_length)
{
frontier[tid]=frontier2[tid];
}
if(tid==0)
{
g_mutex=0;
g_mutex2=0;
*g_q_offsets=0;
*g_q_size=0;
}
}