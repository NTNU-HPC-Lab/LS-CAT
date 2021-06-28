#include "includes.h"
__global__ void sleepKernel(double* cycles, int64_t waitCycles) {
extern __shared__ int s[];
long long int start = clock64();
for (;;) {
auto total = clock64() - start;
if (total >= waitCycles) { break; }
}
*cycles = (double(clock64() - start));
}