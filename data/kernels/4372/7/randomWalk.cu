#include "includes.h"
__global__ void randomWalk(double *results, int T, int N) {
curandState_t state;
curand_init (1234, 0, 0, &state);
double random;

results[0] = 0.0;

for (int j = 1; j < N; j++) {
random = curand_normal_double(&state);
results[j] = results[j-1] + random * sqrt((double) T / N);
}

/*
Generate 2 doubles at once. Test later to see if this is more efficient:
double curand_normal2_double (state);
*/

}