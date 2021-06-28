#include "includes.h"
/*
Copyright [2019] [illava(illava@outlook.com)]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/




const int blockSize = 1024;

// The original code

// d_key is the original key, keeps untouched

// d_temp is a copy of d_key, changes during algorithm

// shift = 2 ^ d

// __global__ void uphill(uint32_t *d_value, uint8_t *d_key, uint8_t *d_temp,
//                        int64_t n, int64_t shift)
// {
//     int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
__global__ void downhill1(uint32_t *d_value, uint8_t *d_key, uint8_t *d_temp, int64_t n, int64_t shift)
{
int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx + shift < n)
{
if (idx % (2 * shift) == 0)
{
uint32_t tmp                   = d_value[n - 1 - (idx + shift)];
d_value[n - 1 - (idx + shift)] = d_value[n - 1 - idx];
if (d_key[n - 1 - (idx + shift - 1)] == 1)
d_value[n - 1 - idx] = 0;
else if (d_temp[n - 1 - (idx + shift)] == 1)
d_value[n - 1 - idx] = tmp;
else
d_value[n - 1 - idx] = tmp + d_value[n - 1 - idx];
d_temp[n - 1 - (idx + shift)] = 0;
}
}
}