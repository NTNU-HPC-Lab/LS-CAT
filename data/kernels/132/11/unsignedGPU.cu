#include "includes.h"
__global__ void unsignedGPU(int numTests, unsigned* ns, unsigned* ds, unsigned* qs, unsigned* rs) {
for (int i = 0; i < numTests; ++i) {
unsigned n = ns[i];
unsigned d = ds[i];

qs[i] = n / d;
rs[i] = n % d;
}
}