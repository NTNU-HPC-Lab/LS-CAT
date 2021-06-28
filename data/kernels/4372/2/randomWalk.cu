#include "includes.h"
__global__ void randomWalk(double *results, int *crossTimes, int T, int N, int numSims, double lowerThreshold, double upperThreshold, int *dev_failCross, double seconds) {
int crossTimeIndex = threadIdx.x + blockIdx.x * blockDim.x;
if (crossTimeIndex < numSims) {
curandState_t state;
curand_init (blockIdx.x * 1000 + threadIdx.x + seconds, 0, 0, &state);
double random;
int start = (threadIdx.x + blockIdx.x * blockDim.x) * N;

crossTimes[crossTimeIndex] = 0;
results[start] = 0.0;
bool crossed = false;

for (int j = start + 1; j < start + N; j++) {
random = curand_normal_double(&state);
results[j] = results[j-1] + random * sqrt((double) T / N);
if (!crossed && results[j] >= upperThreshold) {
crossTimes[crossTimeIndex] = j - start;
crossed = true;
}
else if (!crossed && results[j] <= lowerThreshold) {
crossTimes[crossTimeIndex] = -1 * (j - start);
crossed = true;
}
}

if (!crossed) {
atomicAdd(dev_failCross, 1);
}

}


/*
Generate 2 doubles at once. Test later to see if this is more efficient:
double curand_normal2_double (state);
*/

}