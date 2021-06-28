#include "includes.h"

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

struct Edge{
long long int x;
};



///*
//*/










__global__ void hook_even(int* parent, Edge* edge_list, int e, bool* flag, bool* active_edges){
int bid = blockIdx.x;
int tid = threadIdx.x;
int id = bid*blockDim.x + tid;
long long int x;
int u, v, mx, mn, parent_u, parent_v;
__shared__ bool block_flag;
if(tid == 0)
block_flag = false;
__syncthreads();
if(id < e)
if(active_edges[id]){
x = edge_list[id].x;
v = (int) x & 0xFFFFFFFF;
u = (int) (x >> 32);

parent_u = parent[u];
parent_v = parent[v];

mx = max(parent_u, parent_v);
mn = parent_u + parent_v - mx;

if(parent_u == parent_v)
active_edges[id] = false;
else{
parent[mn] = mx;
block_flag = true;
}
}
__syncthreads();

if(tid == 0)
if(block_flag)
*flag = true;
return;
}