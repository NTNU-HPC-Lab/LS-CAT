#include "includes.h"

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

struct Edge{
long long int x;
};



///*
//*/










__global__ void initialize_parent(int* parent, int n){
int bid = blockIdx.x;
int id = bid*blockDim.x + threadIdx.x;
if(id < n)
parent[id] = id;
return;
}