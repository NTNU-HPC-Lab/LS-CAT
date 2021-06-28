#pragma once
#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "StateAction.h"
#include "Asteroids\Random.h"

enum type{
	DEEP
};

class DeepLearner
{
	type algo;
	Random rand;
	int* width;
	int* height;
	int* score;
	int* inputStorage;
	int* keypressStorage;

	int lastScore;
	int rWidth;
	int rHeight;
	int numCalls;
	int lastInput;
	int numInput;
	int screenStorageCount;
	int firstHiddenWeightsSize;
	int numFirstHiddenNeurons;
	int screenSize;
	int numScreenHistory;

	float* reduceScreen;
	float* inputWeights;

	//256 hidden nodes
	float* firstHiddenWeights;

	//Unused 15 hidden nodes
	float* secondHiddenWeights;

	//Unused 10 hidden nodes
	float* thirdHiddenWeights;
	float lr;
	float f_RandomChance;

	float* InputBias;
	float* firstBias;
	float* secondBias;
	float* outputWeights;

	//Will store the screen as the 400x300 greyscaled image
	float* screenStorage;

	float* FirstLayerFire;
	float* OutputLayerTotals;
	float* FirstLayerStorage;
	float* OutputLayerStorage;

	bool FullStorage;
	void GetScreen();

public:
	bool pause = false;


	int GetInput(std::vector<float*> screengrab);
	void Initialize(int* score, int* widthPoint, int* heightPoint, int numInput, float learningRate, type algoType = type::DEEP);
	void GameOver(bool isWin);
	void SwitchAlgorithm(type algoType);
	void learn(bool isWin);
	void play();
	void StoreScreen(float* screenBits);
	void ResetScore();

	DeepLearner();
	~DeepLearner();
};

//
//#include <iostream>
//#include <algorithm>
//using namespace std;
//#define N 1024
//#define RADIUS 3
//#define BLOCK_SIZE 16
//
//__global__ void stencil_1d(int *in, int *out) {
//	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
//	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
//	int lindex = threadIdx.x + RADIUS;
//	// Read input elements into shared memory
//	temp[lindex] = in[gindex];
//	if (threadIdx.x < RADIUS) {
//		temp[lindex - RADIUS] = in[gindex - RADIUS];
//		temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
//	}
//	// Synchronize (ensure all the data is available)
//	__syncthreads();
//	// Apply the stencil
//	int result = 0;
//	for (int offset = -RADIUS; offset <= RADIUS; offset++)
//		result += temp[lindex + offset];
//	// Store the result
//	out[gindex] = result;
//}
//void fill_ints(int *x, int n) {
//	fill_n(x, n, 1);
//}
//int main(void) {
//	int *in, *out; // host copies of a, b, c
//	int *d_in, *d_out; // device copies of a, b, c
//	int size = (N + 2 * RADIUS) * sizeof(int);
//	// Alloc space for host copies and setup values
//	in = (int *)malloc(size); fill_ints(in, N + 2 * RADIUS);
//	out = (int *)malloc(size); fill_ints(out, N + 2 * RADIUS);
//	// Alloc space for device copies
//	cudaMalloc((void **)&d_in, size);
//	cudaMalloc((void **)&d_out, size);
//	// Copy to device
//	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);
//	// Launch stencil_1d() kernel on GPU
//	stencil_1d << <N / BLOCK_SIZE, BLOCK_SIZE >> >(d_in + RADIUS, d_out + RADIUS);
//	// Copy result back to host
//	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
//	// Cleanup
//	free(in); free(out);
//	cudaFree(d_in); cudaFree(d_out);
//	return 0;
//}
