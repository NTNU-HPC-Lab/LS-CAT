#include "includes.h"
__device__ inline static float euclid_distance(int    numCoords, int    numObjs, int    numClusters, int    tid, int    clusterId, float *objects, float *clusters )
{
float ans=0.0;
for (int i = 0; i < numCoords; i++) {
ans += (objects[3*tid+i] - clusters[i + clusterId*3]) *
(objects[3*tid+i] - clusters[i + clusterId*3]);
}

return(ans);
}
__global__ static void find_nearest_cluster(int numCoords, int numObjs, int numClusters, float *objects, float *deviceClusters, int *membership, int *changedmembership )
{
extern __shared__ float sharedMem[];
float *sh_Clusters = sharedMem;
float *sh_Objects = (float*)&sh_Clusters[numClusters * 3];

for(int i = 0; i < numCoords * numClusters; i++) {
sh_Clusters[i] = deviceClusters[i];
}
__syncthreads();

unsigned int tid = threadIdx.x;
int objectId = blockDim.x * blockIdx.x + threadIdx.x;

while (objectId < numObjs) {
int   index, i;
float dist, min_dist;

for(int i = 0; i < numCoords; i++) {
sh_Objects[3*tid+i] = objects[3*objectId+i];
}

index = 0;
min_dist = euclid_distance(numCoords, numObjs, numClusters, tid,
0, sh_Objects, sh_Clusters);

for (i=1; i<numClusters; i++) {
dist = euclid_distance(numCoords, numObjs, numClusters, tid,
i, sh_Objects, sh_Clusters);
if (dist < min_dist) {
min_dist = dist;
index    = i;
}
}
if (membership[objectId] != index)
{
changedmembership[objectId] = 1;
membership[objectId] = index;

}
objectId += blockDim.x * gridDim.x;
}
}