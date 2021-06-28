#include "includes.h"

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

struct Edge{
long long int x;
};



///*
//*/










__global__ void hook_init(int* parent, Edge* edge_list, int e){
int bid = blockIdx.x;
int id = bid*blockDim.x + threadIdx.x;
long long int x;
int u, v, mx, mn;
if(id < e){
x = edge_list[id].x;
v = (int) x & 0xFFFFFFFF;
u = (int) (x >> 32);

mx = max(u, v);
mn = u + v - mx;
parent[mx] = mn;
}
return;
}