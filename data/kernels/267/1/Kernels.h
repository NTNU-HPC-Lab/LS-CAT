/*
 * Kernels.h
 *
 *  Created on: 22/08/2014
 *      Author: vincentvillani
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include <cuda_runtime.h>
#include <stdio.h>
#include <cuComplex.h>

#include <stdint.h>

__device__ __host__ uint64_t upperTriangularLength(unsigned int numRows);


//Parallel Reduce
//---------------------
//__global__ void parallelReduceUnroll2(float* d_inputArray, uint64_t inputLength, float* d_outputArray);

//Calculates the mean of an input array in parallel, in place in the input array
__global__ void parallelMeanUnroll2(float* d_inputArray, uint64_t inputLength, float* d_outputMean);

//Subtracts mean from the input array
__global__ void subtractMean(float* d_inputArray, uint64_t inputLength, float d_mean);

//---------------------

//Outer products
//---------------------
__global__ void outerProductSmartBruteForce(float* resultMatrix, float* vec, int vectorLength);

__global__ void outerProductSmartBruteForceLessThreads(float* resultMatrix, float* vec, uint64_t vectorLength);

//Specialised outer product for DSPSR
__global__ void outerProductUpperTri(cuFloatComplex* resultMatrix, cuFloatComplex* vec, unsigned int vectorLength);
//---------------------

//Other stuff
__global__ void normalise(float* result, unsigned int resultLength, float* amps, unsigned int* hits);

//Sets the digaonal of the matrix to zero. Assumes square matrices
__global__ void setDiagonalToZero(float* d_matrix, uint64_t columnsAndRows);



#endif /* KERNELS_H_ */
