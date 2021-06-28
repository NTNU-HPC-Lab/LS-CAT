/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * als.h
 *
 *  Created on: Aug 13, 2015
 *      Author: weitan
 */

#ifndef ALS_H_
#define ALS_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <fstream>
#include <sys/time.h>
#include <cusparse.h>
#include <host_defines.h>
#include <cstdlib>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
//these parameters do not change among different problem size
//our kernels handle F=100 only, anyway
//#define F 100
#define SCAN_BATCH 24
#define TILE_SIZE F/10
#define T10 10

#define cudacall(call) \
    do\
    {\
	cudaError_t err = (call);\
	if(cudaSuccess != err)\
	    {\
		fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
		throw thrust::system_error(err, thrust::cuda_category(), cudaGetErrorString(err));\
	    }\
    }\
    while (0)\

#define cublascall(call) \
do\
{\
	cublasStatus_t status = (call);\
	if(CUBLAS_STATUS_SUCCESS != status)\
	{\
		fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);\
		cudaDeviceReset();\
		exit(EXIT_FAILURE);\
	}\
}\
while(0)\

#define cusparsecall(call) \
do\
{\
	cusparseStatus_t status = (call);\
	if(CUSPARSE_STATUS_SUCCESS != status)\
	{\
		fprintf(stderr,"CUSPARSE Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);\
		cudaDeviceReset();\
		exit(EXIT_FAILURE);\
	}\
}\
while(0)\

#define cudaCheckError() {                                          \
        cudaError_t e = cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }\
while(0)\

inline double seconds(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

extern "C" {

void doALSWithCSR(cudaStream_t cuda_stream, int* csrRowIndex, int* csrColIndex, float* csrVal,
		float* thetaTHost, float* XTHost,
		const int m, const int n, const int f, const long nnz, const float lambda,
		const int X_BATCH);

}

#endif /* ALS_H_ */
