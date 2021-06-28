#include "includes.h"
__global__ void BFS_kernel_one_block_spill( volatile unsigned int *frontier, unsigned int frontier_len, volatile unsigned int *cost, volatile int *visited, unsigned int *edgeArray, unsigned int *edgeArrayAux, unsigned int numVertices, unsigned int numEdges, volatile unsigned int *frontier_length, const unsigned int max_local_mem)
{

extern volatile __shared__ unsigned int s_mem[];

//block queues
unsigned int *b_q=(unsigned int *)&s_mem[0];
unsigned int *b_q2=(unsigned int *)&s_mem[max_local_mem];

volatile __shared__ unsigned int b_offset[1];
volatile __shared__ unsigned int b_q_length[1];
//get the threadId
unsigned int tid=threadIdx.x;
//copy frontier queue from global queue to local block queue
if(tid<frontier_len)
{
b_q[tid]=frontier[tid];
}

unsigned int f_len=frontier_len;
while(1)
{
//Initialize the block queue size to 0
if(tid==0)
{
b_q_length[0]=0;
b_offset[0]=0;
}
__syncthreads();
if(tid<f_len)
{
//get the nodes to traverse from block queue
unsigned int node_to_process=*(volatile unsigned int *)&b_q[tid];
//remove from frontier
visited[node_to_process]=0;
//get the offsets of the vertex in the edge list
unsigned int offset = edgeArray[node_to_process];
unsigned int next   = edgeArray[node_to_process+1];

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
unsigned int t=
atomicAdd((unsigned int *)&b_q_length[0],1);
if(t< max_local_mem)
{
b_q2[t]=nid;
}
//write to global memory if shared memory full
else
{
int off=atomicAdd((unsigned int *)&b_offset[0],
1);
frontier[off]=nid;
}
}
}
offset++;
}
}
__syncthreads();

if(tid<max_local_mem)
b_q[tid]=*(volatile unsigned int *)&b_q2[tid];

__syncthreads();
//Traversal complete exit
if(b_q_length[0]==0)
{
if(tid==0)
frontier_length[0]=0;
return;
}
// If frontier exceeds one block in size copy warp queues to
//global frontier queue and exit
else if( b_q_length[0] > blockDim.x || b_q_length[0] > max_local_mem)
{
if(tid<(b_q_length[0]-b_offset[0]))
frontier[b_offset[0]+tid]= *(volatile unsigned int *)&b_q[tid];
if(tid==0)
{
frontier_length[0] = b_q_length[0];
}
return;
}
f_len=b_q_length[0];
__syncthreads();
}
}