#include "includes.h"

using namespace std;

void KNearestNeighborsCPU(float3 *dataArray, int *result, int cnt);

// cpu algorithm
__global__ void KNearestNeighborsGPU(float3 *dataArray, int *result, int cnt)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id >= cnt) return;

float3 point = dataArray[id], current;
float minimumDist = 3.4028234664e38f, distance = 0;

for (int j = 0; j < cnt; j++)
{
if (id == j) continue;

current = dataArray[j];
distance = (point.x - current.x) * (point.x - current.x);
distance += (point.y - current.y) * (point.y - current.y);
distance += (point.z - current.z) * (point.z - current.z);

if (distance < minimumDist)
{
minimumDist = distance;
result[id] = j;
}
}
}