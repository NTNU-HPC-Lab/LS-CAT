#include "includes.h"
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
long i = start;
long j = middle;
for (long k = start; k < end; k++) {
if (i < middle && (j >= end || source[i] < source[j])) {
dest[k] = source[i];
i++;
} else {
dest[k] = source[j];
j++;
}
}
}
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
int x;
return threadIdx.x +
threadIdx.y * (x  = threads->x) +
threadIdx.z * (x *= threads->y) +
blockIdx.x  * (x *= threads->z) +
blockIdx.y  * (x *= blocks->z) +
blockIdx.z  * (x *= blocks->y);
}
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
unsigned int idx = getIdx(threads, blocks);
long start = width*idx*slices,
middle,
end;

for (long slice = 0; slice < slices; slice++) {
if (start >= size)
break;

middle = min(start + (width >> 1), size);
end = min(start + width, size);
gpu_bottomUpMerge(source, dest, start, middle, end);
start += width;
}
}