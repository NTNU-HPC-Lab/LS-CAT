#include "includes.h"

// Subpart A:
// Write step 1 as a kernel that operates on threads 0--31.
// Assume that the input flags are 0 for false and 1 for true and are stored
// in a local per-thread register called p (for predicate).
//
// You have access to 31 words of shared memory s[0:31], with s[0]
// corresponding to thread 0 and s[31] corresponding to thread 31.
// You may change the values of s[0:31]. Put the return sum in s[0].
// Your code should execute no more than 5 warp-wide addition operations.

__device__ unsigned int shared_reduce(unsigned int p, volatile unsigned int * s) {
// Assumes values in 'p' are either 1 or 0
// Assumes s[0:31] are allocated
// Sums p across warp, returning the result. Suggest you put
// result in s[0] and return it
// You may change any value in s
// You should execute no more than 5 + operations (if you're doing
// 31, you're doing it wrong)

int tid = threadIdx.x;
s[tid] = p;
__syncthreads();
for (int i = blockDim.x / 2; i > 0; i >>= 1) { // This could be unrolled
if (tid < i) {
s[tid] += s[tid+i];
}
__syncthreads();
}
return s[0];
}
__global__ void reduce(unsigned int * d_out_shared, const unsigned int * d_in)
{
extern __shared__ unsigned int s[];
int t = threadIdx.x;
int p = d_in[t];
unsigned int sr = shared_reduce(p, s);
if (t == 0)
{
*d_out_shared = sr;
}
}