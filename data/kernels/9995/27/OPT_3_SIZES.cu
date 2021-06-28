#include "includes.h"
__global__ void OPT_3_SIZES(int* adj, int* lcmsizes, int* sizes, int n) {

int vertex = blockIdx.x;
int vcomp = threadIdx.x;
int cval;

if(vertex < n && vcomp < n)
for(int i = vcomp; i < n; i += blockDim.x) {

//skips to next vertex
if(vertex == i) {
continue;
}

//resets count
cval = 0;

//for loop that goes through vertex neighbors
for(int j = 0; j < sizes[vertex + 1] - sizes[vertex]; j++) {

//loop compares to other vertex i/vcomp
for(int k = 0; k < sizes[i+1] - sizes[i]; k++) {

if(adj[sizes[vertex] + j] == adj[sizes[i] + k]) {

++cval;
break;
}
}

if(cval > 0) {
atomicAdd(&lcmsizes[vertex + 1], 1);
break;
}
}
}
}