#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "global_definitions.h"

class TridiagSolver {
public:
	enum class Method {
		THOMAS,
		PCR
	};

	TridiagSolver(int dim, int size, Method type);

	void setParamsLocation(ParamsLocation location);

	void solve(float* a, float* b, float* c, float* y);
	void solve(float a, float b, float c, float* y);

	float getElapsedTime() { return lastElapsedTime; }

private:
	Method method;
	size_t size;
	size_t dim;

	ParamsLocation paramsLocation;

	// performance measures
	cudaEvent_t start, stop;
	float lastElapsedTime;
	float avgElapsedTime;
	size_t nbCalls;
};