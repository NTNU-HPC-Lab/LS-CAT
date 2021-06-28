#include "includes.h"
/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree.
*/



__global__ void _slowKernel(char* ptr, int sz) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
for (; idx < sz; idx += (gridDim.x * blockDim.x)) {
for (int i = 0; i < 100000; ++i) {
ptr[idx] += ptr[(idx + 1007) % sz] + i;
}
}
}