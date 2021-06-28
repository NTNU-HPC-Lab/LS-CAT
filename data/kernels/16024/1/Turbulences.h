#pragma once

#include "cuda.h"
#include "time.h" 
#include "curand_kernel.h"
#include <stdio.h>
#include "math.h"


__global__ void normalizePu(float* P, float* Dir)
{
	int i = blockIdx.x * 512 * 3 + threadIdx.x * 3;
	P[i + 2] = 1.0f - __max(P[i], P[i + 1]);
	P[i + 1] = (1.0f - P[i + 2]) - __min(P[i], P[i + 1]);
	P[i] = (1.0f - P[i + 2]) - P[i + 1];
	for (int j = 0; j < 3; j++) {
		P[i + j] = sqrt(P[i + j]) * ((Dir[i + j] > 0.5f) * 2.0f - 1.0f);
	}
}
__global__ void normalizeEpu(float* Ep, float Eporog, float Ekoef)
{
	int i = blockIdx.x * 512 + threadIdx.x;
	Ep[i] = Eporog + Ekoef * Ep[i];
}
__global__ void setup_kernelu(curandState* state, unsigned long seed)
{
	int i = blockIdx.x * 512 + threadIdx.x;
	curand_init(seed, i, 0, &state[i]);
}

__global__ void generateu(curandState* globalState, float* randomArray)
{
	int ind = blockIdx.x * 512 + threadIdx.x;
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);
	randomArray[ind] = RANDOM;
	globalState[ind] = localState;
}

void RandomGenQu(float* devRandomValues, const int N, bool printEnable, unsigned int n) {
	dim3 tpb(N / 512, 1, 1);
	curandState* devStates;
	float* randomValues = new float[N];

	cudaMalloc(&devStates, N * sizeof(curandState));

	// setup seeds
	//printf("%i\n", N);
	setup_kernelu << <tpb, 512 >> > (devStates, time(NULL) + n);

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));// generate random numbers
	generateu << <tpb, 512 >> > (devStates, devRandomValues);

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

void RandomGenEpu(float* devRandomValues, const int N, bool printEnable, unsigned int n, float Eporog, float Ekoef) {
	dim3 tpb(N / 512, 1, 1);
	curandState* devStates;
	float* randomValues = new float[N];

	cudaMalloc(&devStates, N * sizeof(curandState));

	// setup seeds
	//printf("%i\n", N);
	setup_kernelu << <tpb, 512 >> > (devStates, time(NULL) + n);

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));// generate random numbers
	generateu << <tpb, 512 >> > (devStates, devRandomValues);
	normalizeEpu << < tpb, 512 >> > (devRandomValues, Eporog, Ekoef);
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
__global__ void ConstEu(float* E)
{
	int i = blockIdx.x * 512 + threadIdx.x;
	E[i] = 100.0f;
}
void RandomGenPu(float* devRandomValues, const int N, bool printEnable, unsigned int n) {
	dim3 tpb(N / 512, 1, 1);
	curandState* devStates;
	curandState* devStatesDir;
	float* randomValues = new float[N];
	float* randomDir;

	cudaMalloc(&randomDir, N * sizeof(float));
	cudaMalloc(&devStates, N * sizeof(curandState));
	cudaMalloc(&devStatesDir, N * sizeof(curandState));

	// setup seeds
	//printf("%i\n", N);
	setup_kernelu << <tpb, 512 >> > (devStates, time(NULL));
	setup_kernelu << <tpb, 512 >> > (devStatesDir, time(NULL));

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));// generate random numbers
	generateu << <tpb, 512 >> > (devStates, devRandomValues);
	generateu << <tpb, 512 >> > (devStatesDir, randomDir);
	dim3 gtp(N / 512 / 3, 1, 1);
	normalizePu << <gtp, 512 >> > (devRandomValues, randomDir);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	if (printEnable) {
		cudaMemcpy(randomValues, devRandomValues, N * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < N; i += 3)
		{
			printf("%f\n", randomValues[i] * randomValues[i] + randomValues[i + 1] * randomValues[i + 1] + randomValues[i + 2] * randomValues[i + 2]);
		}
	}
	cudaFree(devStates);
	cudaFree(randomDir);
	cudaFree(devStatesDir);

	delete randomValues;
}