#include "includes.h"
__global__ void FindMinSample(float* DistanceBuffer, short2* IndexBuffer, int spread, int mapSizeX, int mapSizeY)
{
int kOffset = CUDASTDOFFSET;
float distance1 = DistanceBuffer[kOffset];
float distance2 = DistanceBuffer[kOffset + spread];
short2 index1 = IndexBuffer[kOffset];
short2 index2 = IndexBuffer[kOffset + spread];

if (kOffset + spread < mapSizeX * mapSizeY)
{
DistanceBuffer[kOffset] = (distance1 < distance2) ? distance1 : distance2;
IndexBuffer[kOffset] = (distance1 < distance2) ? index1 : index2;
}
}