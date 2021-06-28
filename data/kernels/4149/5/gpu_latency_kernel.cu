#include "includes.h"
__global__ void gpu_latency_kernel(int *buffer, size_t reps, size_t steps, size_t elements)
{
int errors = 0;
// this is done with just a single thread
for(size_t j = 0; j < reps; j++) {
int p = j & 31;

for(size_t i = 0; i < steps; i++) {
int next = buffer[p];

if((next >= 0) && (next < elements)) {
p = next;
} else {
printf("%d -> %d\n", p, next);
p = 0;
errors++;
}
}
}
if((errors > 0) && (reps > elements))
buffer[0] = errors;
}