#include "includes.h"
__global__ void _GPU_Floyd_kernel(int k, int *G,int *P, int N){//G will be the adjacency matrix, P will be path matrix
int col=blockIdx.x*blockDim.x + threadIdx.x;
if(col>=N)return;
int idx=N*blockIdx.y+col;

__shared__ int best;
if(threadIdx.x==0)
best=G[N*blockIdx.y+k];
__syncthreads();
if(best==INF || best > 10)return;
int tmp_b=G[k*N+col];
if(tmp_b==INF || tmp_b > 10)return;
//	if (cur > 1)
//		return;
int cur = best + tmp_b;
if(cur<G[idx]){
G[idx]=cur;
P[idx]=k;
}
}