#include "includes.h"
__global__ void ApplyEyeMovement(float* currentEye, float* stats, float moveFactor, float scaleFactor, float scaleBase)
{
float sumWeights = stats[4];

if (sumWeights > 0)
{
currentEye[0] = fmaxf(fminf(moveFactor * stats[0], 1), -1);
currentEye[1] = fmaxf(fminf(moveFactor * stats[1], 1), -1);

float variance = sqrtf((stats[2] + stats[3]) * 0.5);

currentEye[2] = fmaxf(fminf(variance * scaleFactor + scaleBase, 1), 0);
}
else
{
currentEye[0] = 0;
currentEye[1] = 0;
currentEye[2] = 1;
}
}