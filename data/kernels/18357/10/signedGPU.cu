#include "includes.h"
__global__ void signedGPU(int numTests, int* ns, int* ds, int* qs, int* rs) {
for (int i = 0; i < numTests; ++i) {
int n = ns[i];
int d = ds[i];

qs[i] = n / d;
rs[i] = n % d;
}
}