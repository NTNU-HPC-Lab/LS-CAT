#include "includes.h"

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

struct Edge{
long long int x;
};



///*
//*/










__global__ void process_cross_edges(int* parent, Edge* edge_list, int e, bool* flag, bool* cross_edges){
int bid = blockIdx.x;
int tid = threadIdx.x;
int id = bid*blockDim.x + tid;
long long int x;
int u, v, mn, mx, parent_u, parent_v;
__shared__ bool block_flag;
if(tid == 0)
block_flag = false;
__syncthreads();
if(id < e)
if(cross_edges[id]){
x = edge_list[id].x;
v = (int) x & 0xFFFFFFFF;
u = (int) (x >> 32);

parent_u = parent[u];
parent_v = parent[v];

mn = min(parent_u, parent_v);
mx = parent_u + parent_v - mn;

if(parent_u == parent_v)
cross_edges[id] = false;
else{
parent[mx] = mn;
block_flag = true;
}
}
__syncthreads();

if(tid == 0)
if(block_flag)
*flag = true;
return;
}