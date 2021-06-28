#include "includes.h"
__global__ void refreshClusters(dim3 *sum, dim3 *cluster, int *counter)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(counter[i] != 0) {
cluster[i].x = sum[i].x / counter[i];
cluster[i].y = sum[i].y / counter[i];
cluster[i].z = sum[i].z / counter[i];
} else {
cluster[i].z = cluster[i].x = cluster[i].z = 0;
}
sum[i] = dim3(0, 0, 0);
counter[i] = 0;
}