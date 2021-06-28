#include "includes.h"
__global__ void calculateIntermediates(int n, double *xs, int *cluster_index, int *intermediates0, double *intermediates1, double *intermediates2, int k, int d){


int blocksize = n / 450 + 1;
int start = blockIdx.x * blocksize;
int end1 = start + blocksize;
int end;
if (end1>n) end = n;
else end = end1;

if (end > n ) return;
// loop for every K
for (int clust = threadIdx.y; clust < k; clust+= blockDim.y){
// loop for every dimension(features)
for (int dim = threadIdx.x; dim < d; dim+= blockDim.x) {

// Calculate intermediate S0
// for counts we don't have dimensions
if (dim ==0) {
int count = 0;
for(int z=start; z<end; z++)
{
if(cluster_index[z] == clust) {
count ++;
}
}
intermediates0[blockIdx.x*k+clust] = count;
}

// Calculate intermediate S1 and S2
double sum1 = 0.0;
double sum2 = 0.0;
int idx ;
for (int z=start; z<end; z++) {
if(cluster_index[z] == clust) {
idx = z * d + dim;
sum1 += xs[idx];
sum2 += xs[idx] * xs[idx];

}
}
int index = (blockIdx.x*k*d + clust*d + dim);
intermediates1[index] = sum1;
intermediates2[index] = sum2;
}
}
}