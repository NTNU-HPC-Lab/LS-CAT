#include "includes.h"
__global__ void BFS_kernel_multi_block_spill( volatile unsigned int *frontier, volatile unsigned int *frontier2, unsigned int frontier_len, volatile unsigned int *cost, volatile int *visited, unsigned int *edgeArray, unsigned int *edgeArrayAux, unsigned int numVertices, unsigned int numEdges, volatile unsigned int *frontier_length, const unsigned int max_local_mem)
{

extern volatile __shared__ unsigned int b_q[];

volatile __shared__ unsigned int b_q_length[1];
volatile __shared__ unsigned int b_offset[1];
//get the threadId
unsigned int tid=threadIdx.x + blockDim.x * blockIdx.x;
unsigned int lid=threadIdx.x;

//initialize the block queue length and warp queue offset
if (lid == 0 )
{
b_q_length[0]=0;
b_offset[0]=0;
}

__syncthreads();
//Initialize the warp queue sizes to 0
if(tid<frontier_len)
{
//get the nodes to traverse from block queue
unsigned int node_to_process=frontier[tid];
visited[node_to_process]=0;
//get the offsets of the vertex in the edge list
unsigned int offset=edgeArray[node_to_process];
unsigned int next=edgeArray[node_to_process+1];

//Iterate through the neighbors of the vertex
while(offset<next)
{
//get neighbor
unsigned int nid=edgeArrayAux[offset];
//get its cost
unsigned int v=atomicMin((unsigned int *)&cost[nid],
cost[node_to_process]+1);
//if cost is less than previously set add to frontier
if(v>cost[node_to_process]+1)
{
int is_in_frontier=atomicExch((int *)&visited[nid],1);
//if node already in frontier do nothing
if(is_in_frontier==0)
{
//increment the warp queue size
unsigned int t=atomicAdd((unsigned int *)&b_q_length[0],
1);
if(t<max_local_mem)
{
b_q[t]=nid;
}
//write to global memory if shared memory full
else
{
int off=atomicAdd((unsigned int *)frontier_length,
1);
frontier2[off]=nid;
}
}
}
offset++;
}
}

__syncthreads();

//get block queue offset in global queue
if(lid==0)
{
if(b_q_length[0] > max_local_mem)
{
b_q_length[0]=max_local_mem;
}
b_offset[0]=atomicAdd((unsigned int *)frontier_length,b_q_length[0]);
}
__syncthreads();

//copy block queue to frontier
if(lid < b_q_length[0])
frontier2[lid+b_offset[0]]=b_q[lid];
}