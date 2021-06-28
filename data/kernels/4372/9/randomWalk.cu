#include "includes.h"
__global__ void randomWalk(double *results, int *crossTimes, int T, int N, int numSims, double upperThreshold, double deviceID) {

// a variable to keep track of this simulation's position in the crossTimes array
int crossTimeIndex = threadIdx.x + blockIdx.x * blockDim.x;

if (crossTimeIndex < numSims) {

// create random number generator
curandState_t state;
curand_init (blockIdx.x * (1000 * deviceID) + threadIdx.x + clock64(), 0, 0, &state);
double random;

// starting position of this siulation in results array
int start = (threadIdx.x + blockIdx.x * blockDim.x) * N;

// set default value of cross time for this simulation to 0, since the simulation hasn't crossed the threshold yet
crossTimes[crossTimeIndex] = 0;

// starting point of path is 0
results[start] = 0.0;

// boolean to keep track of whether this path has crossed
bool crossed = false;

for (int j = start + 1; j < start + N; j++) {
// generate random number
random = curand_normal_double(&state);

//calculate next step of path
results[j] = results[j-1] + random * sqrt((double) T / N);

// store crossing time as positive value if it has crossed the upper threshold. Negative value if crossed the lower threshold
if (!crossed && results[j] >= upperThreshold) {
crossTimes[crossTimeIndex] = j - start;
crossed = true;
}
}

}

}