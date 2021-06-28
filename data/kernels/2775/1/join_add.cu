#include "includes.h"


namespace {

}  // namespace

__global__ void join_add(const int *d1, const int *d2, int *d3) { d3[0] = d1[0] + d2[0]; }