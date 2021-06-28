#include "includes.h"
extern "C" {

#ifndef REAL
#define REAL float
#endif





















}
__global__ void ge_equals_no_transp (const int sd, const int fd, const REAL* a, const int offset_a, const int ld_a, const REAL* b, const int offset_b, const int ld_b, int* eq_flag) {
const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
const bool valid = (gid_0 < sd) && (gid_1 < fd);
if (valid) {
const int ia = offset_a + gid_0 + gid_1 * ld_a;
const int ib = offset_b + gid_0 + gid_1 * ld_b;
if (a[ia] != b[ib]){
eq_flag[0]++;
}
}
}