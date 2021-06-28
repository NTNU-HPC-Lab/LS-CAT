#include "includes.h"

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

struct Edge{
long long int x;
};



///*
//*/










__global__ void leaf_pointer_jumping(int* parent, int* vertex_state, int n){
int bid = blockIdx.x;
int id = bid*blockDim.x + threadIdx.x;
int parent_id, grandparent_id;
if(id < n)
if(vertex_state[id] == 1){
parent_id = parent[id];
grandparent_id = parent[parent_id];
parent[id] = grandparent_id;
}
return;
}