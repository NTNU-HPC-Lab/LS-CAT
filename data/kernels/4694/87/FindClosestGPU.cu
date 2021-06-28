#include "includes.h"
__global__ void FindClosestGPU(float3* points, int* indices, int count)
{
if (count <= 1) return;
int idx = threadIdx.x + blockIdx.x * blockDim.x;
if (idx < count)
{
float3 thisPoint = points[idx]; // every thread takes its own point
float smallestDistSoFar = 3.40282e38f; // almost the biggest possible floating point value
int smallestIdxSoFar = -1;

// run through the list of all other points
for (int i = 0; i < count; i++)
{
if (i == idx) continue;
float dist = (thisPoint.x - points[i].x)*(thisPoint.x - points[i].x);
dist += (thisPoint.y - points[i].y)*(thisPoint.y - points[i].y);
dist += (thisPoint.z - points[i].z)*(thisPoint.z - points[i].z);
if (dist < smallestDistSoFar)
{
smallestDistSoFar = dist;
smallestIdxSoFar = i;
}
}
indices[idx] = smallestIdxSoFar;
}
}