#include "includes.h"

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

struct Edge{
long long int x;
};



///*
//*/










__global__ void initialize_active_edges(bool* active_edges, int e){
int bid = blockIdx.x;
int id = bid*blockDim.x + threadIdx.x;
if(id < e)
active_edges[id] = true;
return;
}