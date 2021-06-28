#include "includes.h"

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

struct Edge{
long long int x;
};



///*
//*/










__global__ void update_states(int* parent, int* vertex_state, int n){
int bid = blockIdx.x;
int id = bid*blockDim.x + threadIdx.x;
if(id < n)
vertex_state[id] = parent[id] == id ? 0 : 1;
return;
}