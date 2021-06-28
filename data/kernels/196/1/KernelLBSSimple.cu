#include "includes.h"
__global__ void KernelLBSSimple(int aCount, const int* b_global, int bCount, int* indices_global) {

__shared__ int data_shared[NT * VT];

int tid = threadIdx.x;

// Load bCount elements from B into data_shared.
int x[VT];
#pragma unroll
for(int i = 0; i < VT; ++i) {
int index = NT * i + tid;
if(index < bCount) x[i] = b_global[index];
}

#pragma unroll
for(int i = 0; i < VT; ++i)
data_shared[NT * i + tid] = x[i];
__syncthreads();

// Each thread searches for its Merge Path partition.
int diag = VT * tid;
int begin = max(0, diag - bCount);
int end = min(diag, aCount);

while(begin < end) {
int mid = (begin + end)>> 1;
int aKey = mid;
int bKey = data_shared[diag - 1 - mid];
bool pred = aKey < bKey;
if(pred) begin = mid + 1;
else end = mid;
}
int mp = begin;

// Sequentially search, comparing indices a to elements data_shared[b].
// Store indices for A in the right-side of the shared memory array.
// This lets us complete the search in just a single pass, rather than
// the search and compact passes of the generalized vectorized sorted
// search function.
int a = mp;
int b = diag - a;

#pragma unroll
for(int i = 0; i < VT; ++i) {
bool p;
if(b >= bCount) p = true;
else if(a >= aCount) p = false;
else p = a < data_shared[b];

if(p)
// If a < data_shared[b], advance A and store the index b - 1.
data_shared[bCount + a++] = b - 1;
else
// Just advance b.
++b;
}
__syncthreads();

// Store all indices to global memory.
for(int i = tid; i < aCount; i += NT)
indices_global[i] = data_shared[bCount + i];
}