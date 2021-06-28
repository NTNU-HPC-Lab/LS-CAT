#include "includes.h"
extern "C" {

#ifndef REAL
#define REAL float
#endif





















}
__global__ void ge_set (const int sd, const int fd, const REAL val, REAL* a, const int offset_a, const int ld_a) {
const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
const bool valid = (gid_0 < sd) && (gid_1 < fd);
if (valid) {
a[offset_a + gid_0 + gid_1 * ld_a] = val;
}
}