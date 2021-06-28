#include "includes.h"
/*------------------------GPU RANKING----------------------------------------START-------*/

/*------------------------shfl_scan_test-----------------------------------------------Start*/
/*------------------------shfl_scan_test-----------------------------------------------End*/
/*------------------------Final Ranking-----------------------------------------------Start*/

/*------------------------Final_ranking-----------------------------------------------End*/

/*-----------------------GPU RANKING------------------------------------------END--------*/

/*-----------------------iDivUp--------------------------------------------------------Start*/

__global__ void final_ranking(float *data , int *rank , float *partial_data , int *partial_rank , int len)
{
__shared__ float value_buf;
__shared__ int rank_buf;

int id = ((blockIdx.x*blockDim.x)+threadIdx.x);
if(id>len) return;

if(threadIdx.x == 0)
{
value_buf = partial_data[blockIdx.x];
rank_buf = partial_rank[blockIdx.x];
}
__syncthreads();
if(data[id] == value_buf)
{
rank[id] = rank_buf;
}
}