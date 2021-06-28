#include "includes.h"

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

struct Edge{
long long int x;
};



///*
//*/










__global__ void accumulate(Edge* edge_list, bool* cross_edges, int* indices, int e){
int bid = blockIdx.x;
int id = bid*blockDim.x + threadIdx.x;
Edge temp;
temp.x = 0;
if(id < e)
if(cross_edges[id])
temp = edge_list[id];
__syncthreads();
if(temp.x)
edge_list[indices[id]] = temp;
return;
}