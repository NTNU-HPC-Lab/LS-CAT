#include "includes.h"
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#define EIGEN_USE_GPU

#ifdef HIGH_PREC
typedef double  VALUETYPE;
#else
typedef float   VALUETYPE;
#endif

typedef unsigned long long int_64;

__global__ void get_i_idx_se_a(const int nloc, const int * ilist, int * i_idx)
{
const unsigned int idy = blockIdx.x * blockDim.x + threadIdx.x;
if(idy >= nloc) {
return;
}
i_idx[ilist[idy]] = idy;
}