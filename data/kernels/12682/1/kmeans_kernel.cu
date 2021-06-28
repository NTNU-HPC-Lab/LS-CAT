#include "includes.h"
__global__ void kmeans_kernel(const sequence_t *data, const sequence_t *centroids, int * membership, unsigned int * tmp_centroidCount, unsigned int n, unsigned int numClusters )
{

int index = blockIdx.x * blockDim.x  + threadIdx.x;
if (index < n){

unsigned int min_distance = UINT_MAX;
long nearest = -1;

for (int i = 0; i < numClusters; i++){
sequence_t centroid = centroids[i];
unsigned int distance = __popcll(centroid.x ^ data[index].x) +
__popcll(centroid.y ^ data[index].y) +
__popcll(centroid.z ^ data[index].z);
if(distance < min_distance) {
nearest = i;
min_distance = distance;
}
}

if(membership[index] != nearest) {
membership[index]=nearest;
atomicInc(&cuda_delta,n*2);
}

unsigned int *tmp_centroid = &tmp_centroidCount[membership[index] * BIT_SIZE_OF(sequence_t)];
for (unsigned int j=0;j<SEQ_DIM_BITS_SIZE;j++)  {
// bits tmp_centroid[0] is less significative bit from sequence_t
// bits tmp_centroid[0] = z << 0
unsigned long int mask = 1;
if (data[index].z & (mask << j)){
atomicInc(&tmp_centroid[j],n);
}
if (data[index].y & (mask << j)){
atomicInc(&tmp_centroid[SEQ_DIM_BITS_SIZE + j],n);
}
if (data[index].x & (mask << j)){
atomicInc(&tmp_centroid[(2 *SEQ_DIM_BITS_SIZE)+j],n);
}
}
}
}