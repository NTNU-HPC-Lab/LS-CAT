#include "includes.h"
__global__ void FindClosestPoint(float3 *points, int *closestPoint, const int numberPoints)
{
// used to identify the thread that is currently running
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// now find the closest point to each point
// 'i' represents the current point that we are finding the closest point to!
int distanceBetweenPoints = 9999999, tempDistance = 0;
for (int j = 0; j < numberPoints; j++)
if (idx != j)		// dont check the distance between the point and itself
{
tempDistance = pow((points[idx].x - points[j].x), 2) + pow((points[idx].y - points[j].y), 2);
if (tempDistance < distanceBetweenPoints)
{
distanceBetweenPoints = tempDistance;
closestPoint[idx] = j;
}
}
}