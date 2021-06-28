#include "includes.h"

#define N 10000000
#define MAX_ERR 1e-6


__global__ void vector_add(float *out, float *a, float *b, int n) {
for(int i = 0; i < n; i ++){
out[i] = a[i] + b[i];
}
}