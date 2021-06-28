#include "includes.h"
__global__ void checkAggregationFillAggregates(int size, int *adjIndices, int *adjacency, int* aggregation, int* valuesIn, int* valuesOut, int* incomplete) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size)
{
// Find the currently marked distance
int currentVal = valuesIn[idx];
int currentAgg = aggregation[idx];

// Checking if any neighbors have a better value
int start = adjIndices[idx];
int end = adjIndices[idx + 1];
for (int i = start; i < end; i++)
{
int neighborAgg = aggregation[adjacency[i]];
int neighborVal = valuesIn[adjacency[i]];
if (neighborAgg == currentAgg && neighborVal > currentVal)
{
currentVal = neighborVal;
incomplete[0] = 1;
}
}

// Write out the distance to the output vector:
valuesOut[idx] = currentVal;
}
}