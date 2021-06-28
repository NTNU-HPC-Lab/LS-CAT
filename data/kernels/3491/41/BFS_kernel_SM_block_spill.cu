#include "includes.h"
__device__ void __gpu_sync(int blocks_to_synch)
{
__syncthreads();
//thread ID in a block
int tid_in_block= threadIdx.x;


// only thread 0 is used for synchronization
if (tid_in_block == 0)
{
atomicAdd((int *)&g_mutex, 1);
//only when all blocks add 1 to g_mutex will
//g_mutex equal to blocks_to_synch
while(g_mutex < blocks_to_synch);
}
__syncthreads();
}
__global__ void BFS_kernel_SM_block_spill( volatile unsigned int *frontier, volatile unsigned int *frontier2, unsigned int frontier_len, volatile unsigned int *cost, volatile int *visited, unsigned int *edgeArray, unsigned int *edgeArrayAux, unsigned int numVertices, unsigned int numEdges, volatile unsigned int *frontier_length, const unsigned int max_local_mem)
{
extern volatile __shared__ unsigned int b_q[];

volatile __shared__ unsigned int b_q_length[1];
volatile __shared__ unsigned int b_offset[1];

//get the threadId
unsigned int tid=threadIdx.x + blockDim.x * blockIdx.x;
unsigned int lid=threadIdx.x;

int loop_index=0;
unsigned int l_mutex=g_mutex2;
unsigned int f_len=frontier_len;
while(1)
{
//initialize the block queue length and warp queue offset
if (lid==0)
{
b_q_length[0]=0;
b_offset[0]=0;
}
__syncthreads();
//Initialize the warp queue sizes to 0
if(tid<f_len)
{
//get the nodes to traverse from block queue
unsigned int node_to_process;

if(loop_index==0)
node_to_process=frontier[tid];
else
node_to_process=frontier2[tid];

//remove from frontier
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
int off=atomicAdd((unsigned int *)g_q_offsets,1);
if(loop_index==0)
frontier2[off]=nid;
else
frontier[off]=nid;
}
}
}
offset++;
}
}
//get offset of block queue in global queue
__syncthreads();
if(lid==0)
{
if(b_q_length[0] > max_local_mem)
{
b_q_length[0] = max_local_mem;
}
b_offset[0]=atomicAdd((unsigned int *)g_q_offsets,b_q_length[0]);
}
__syncthreads();

l_mutex+=gridDim.x;
__gpu_sync(l_mutex);

//store frontier size
if(tid==0)
{
g_q_size[0]=g_q_offsets[0];
g_q_offsets[0]=0;
}

//copy block queue to global queue
if(lid < b_q_length[0])
{
if(loop_index==0)
frontier2[lid+b_offset[0]]=b_q[lid];
else
frontier[lid+b_offset[0]]=b_q[lid];
}

l_mutex+=gridDim.x;
__gpu_sync(l_mutex);

//if frontier exceeds SM blocks or less than 1 block exit
if(g_q_size[0] < blockDim.x ||
g_q_size[0] > blockDim.x * gridDim.x)
{

//TODO:Call the 1-block bfs right here
break;
}
loop_index=(loop_index+1)%2;
//store the current frontier size
f_len=g_q_size[0];
}

if(loop_index==0)
{
for(int i=tid;i<g_q_size[0];i += blockDim.x*gridDim.x)
frontier[i]=frontier2[i];
}

if(tid==0)
{
frontier_length[0]=g_q_size[0];
}
}