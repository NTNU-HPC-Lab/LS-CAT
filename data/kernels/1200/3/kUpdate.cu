#include "includes.h"
__global__ void kUpdate(int nbSpx, float* clusters, float* accAtt_g)
{
int cluster_idx = blockIdx.x*blockDim.x + threadIdx.x;

if (cluster_idx<nbSpx){
int nbSpx2 = nbSpx * 2;
int nbSpx3 = nbSpx * 3;
int nbSpx4 = nbSpx * 4;
int nbSpx5 = nbSpx * 5;
int counter = accAtt_g[cluster_idx + nbSpx5];
if (counter != 0){
clusters[cluster_idx] = accAtt_g[cluster_idx] / counter;
clusters[cluster_idx + nbSpx] = accAtt_g[cluster_idx + nbSpx] / counter;
clusters[cluster_idx + nbSpx2] = accAtt_g[cluster_idx + nbSpx2] / counter;
clusters[cluster_idx + nbSpx3] = accAtt_g[cluster_idx + nbSpx3] / counter;
clusters[cluster_idx + nbSpx4] = accAtt_g[cluster_idx + nbSpx4] / counter;

//reset accumulator
accAtt_g[cluster_idx] = 0;
accAtt_g[cluster_idx + nbSpx] = 0;
accAtt_g[cluster_idx + nbSpx2] = 0;
accAtt_g[cluster_idx + nbSpx3] = 0;
accAtt_g[cluster_idx + nbSpx4] = 0;
accAtt_g[cluster_idx + nbSpx5] = 0;
}
}
}