#include "includes.h"
/**
#Copyright 2013 Athanassios Kintsakis

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

# Author: Athanassios Kintsakis
# contact: akintsakis@issel.ee.auth.gr, athanassios.kintsakis@gmail.com
**/
#define inf 9999





__global__ void funct2(int n, int k, float* x, int* qx) {
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int j = ix & (n - 1);
float temp2 = x[ix - j + k] + x[k * n + j];
if (x[ix] > temp2) {
x[ix] = temp2;
qx[ix] = k;
}
}