#include "includes.h"
/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

/**********************************
When updating a kernel or adding a new one,
please compile the ptx file and commit it:
nvcc -ptx -arch=sm_30 SystemML.cu
***********************************/



/**
* Does a copy of upper to lower triangle of the given matrix
* @param ret the input and output array allocated on the GPU
* @param dim the number of rows of the square matrix ret
* @param N total number of elements of the matrix
*/
extern "C"

extern "C"
__global__ void matrix_atan(double *A, double *C, unsigned int size) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < size){
C[index] = atan(A[index]);
}
}