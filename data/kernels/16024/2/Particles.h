#pragma once

#include "cuda.h"
#include "time.h" 
#include "curand_kernel.h"
#include <stdio.h>
#include "math.h"


__global__ void normalizeP(float* P, float* Dir)
{
	int i = blockIdx.x * 3;
	P[i + 2] = 1 - __max(P[i], P[i + 1]);
	P[i + 1] = (1 - P[i + 2 ]) - __min(P[i], P[i + 1 ]);
	P[i] = (1 - P[i + 2 ]) - P[i + 1];
	for (int j = 0; j < 3; j++) {
		P[i + j ] = sqrt(P[i + j ]) * ((Dir[i + j] > 0.5) * 2 - 1);
	}
}
__global__ void normalizeEp(float* Ep, float Eporog, float Ekoef)
{
	int i = blockIdx.x;
	Ep[i] = Eporog + Ekoef * Ep[i];
}
__global__ void setup_kernel(curandState* state, unsigned long seed)
{
	int id = blockIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

__global__ void generate(curandState* globalState, float* randomArray)
{
	int ind = blockIdx.x;
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);
	randomArray[ind] = RANDOM;
	globalState[ind] = localState;
}

void RandomGenQ(float* devRandomValues, const int N, bool printEnable, unsigned int n) {
	dim3 tpb(N, 1, 1);
	curandState* devStates;
	float* randomValues = new float[N];

	cudaMalloc(&devStates, N * sizeof(curandState));

	// setup seeds
	//printf("%i\n", N);
	setup_kernel << <tpb, 1 >> > (devStates, time(NULL) + n);

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));// generate random numbers
	generate << <tpb, 1 >> > (devStates, devRandomValues);

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	if (printEnable) {
		cudaMemcpy(randomValues, devRandomValues, N * sizeof(*randomValues), cudaMemcpyDeviceToHost);
		for (int i = 0; i < N; i++)
		{
			printf("%f\n", randomValues[i]);
		}
	}
	cudaFree(devStates);
	delete randomValues;
}

void RandomGenEp(float* devRandomValues, const int N, bool printEnable, unsigned int n, float Eporog, float Ekoef) {
	dim3 tpb(N, 1, 1);
	curandState* devStates;
	float* randomValues = new float[N];

	cudaMalloc(&devStates, N * sizeof(curandState));

	// setup seeds
	//printf("%i\n", N);
	setup_kernel << <tpb, 1 >> > (devStates, time(NULL) + n);

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));// generate random numbers
	generate << <tpb, 1 >> > (devStates, devRandomValues);
	normalizeEp << < tpb, 1 >> > (devRandomValues, Eporog, Ekoef);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	if (printEnable) {
		cudaMemcpy(randomValues, devRandomValues, N * sizeof(*randomValues), cudaMemcpyDeviceToHost);
		for (int i = 0; i < N; i++)
		{
			printf("%f\n", randomValues[i]);
		}
	}
	cudaFree(devStates);
	delete randomValues;
}
__global__ void ConstE(float* E)
{
	int i = blockIdx.x;
	E[i] = 1;
}
void RandomGenP(float* devRandomValues, const int N, bool printEnable, unsigned int n) {
	dim3 tpb(N, 1, 1);
	curandState* devStates;
	curandState* devStatesDir;
	float* randomValues = new float[N];
	float* randomDir;

	cudaMalloc(&randomDir, N * sizeof(float));
	cudaMalloc(&devStates, N * sizeof(curandState));
	cudaMalloc(&devStatesDir, N * sizeof(curandState));

	// setup seeds
	//printf("%i\n", N);
	setup_kernel << <tpb, 1 >> > (devStates, time(NULL));
	setup_kernel << <tpb, 1 >> > (devStatesDir, time(NULL));

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));// generate random numbers
	generate << <tpb, 1 >> > (devStates, devRandomValues);
	generate << <tpb, 1 >> > (devStatesDir, randomDir);
	dim3 gtp(N / 3, 1, 1);
	normalizeP << <gtp, 1 >> > (devRandomValues, randomDir);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	if (printEnable) {
		cudaMemcpy(randomValues, devRandomValues, N * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < N; i+=3)
		{
			printf("%f\n", randomValues[i] * randomValues[i] + randomValues[i +1]* randomValues[i +1] + randomValues[i +2]* randomValues[i +2]);
		}
	}
	cudaFree(devStates);
	cudaFree(randomDir);
	cudaFree(devStatesDir);

	delete randomValues;
}