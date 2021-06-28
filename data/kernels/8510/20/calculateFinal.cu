#include "includes.h"
__global__ void calculateFinal(int n, int *intermediates0, double *intermediates1, double *intermediates2, int *s0, double *s1, double *s2, int k, int d){

if (blockIdx.x > 0) return;

// Only block is invoked.
// loop for every K
for (int clust = threadIdx.y; clust < k; clust+= blockDim.y){
// loop for every dimension(features)
for (int dim = threadIdx.x; dim < d; dim+= blockDim.x) {

// Calculate  S0
// for counts we don't have dimensions
if (dim == 0) {
//count = 0;
for(int z = clust; z < 450*k; z+=k){
{
s0[clust] += intermediates0[z];
}
}
}

// Calculate S1 and S2
int start = clust * d + dim;
int kd    = k * d;
double *s1end = &intermediates1[450 * kd];
double *s1cur = &intermediates1[start];
double *s2cur = &intermediates2[start];

for (; s1cur < s1end; s1cur += kd, s2cur += kd)
{
s1[start] += *s1cur;
s2[start] += *s2cur;
}
}
}
}