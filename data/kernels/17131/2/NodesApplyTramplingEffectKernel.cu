#include "includes.h"
__global__ void NodesApplyTramplingEffectKernel(float* target, float* distanceToPath, int graphW, int graphH, float pathThickness, float tramplingCoefficient)
{
int i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
if(i <= graphW && j <= graphH) {
int index = i + j * (graphW + 2);

float t = distanceToPath[index];
t = max(0.0f, min(1.0f, fabsf(t / pathThickness)));
t = t * (t * (-4 * t + 6) - 3) + 1;		// cubic parabola

atomicAdd(&target[index], t * tramplingCoefficient);
}
}