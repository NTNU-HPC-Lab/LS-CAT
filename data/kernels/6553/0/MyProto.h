#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"




cudaError_t updateWeightsWithCuda(double* weights, double* parameters, double* alpha, int* sign, int dimensionSize);