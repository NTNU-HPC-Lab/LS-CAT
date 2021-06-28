#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

/*
*	This class contains functions used to execute 
*	the diamond-square algorithm with CUDA
*/
class CudaAlgorithm
{
public:
	static void CudaDiamondSquare(uint8_t* matrix,
		int matrixSize, int randomValue);
private:
	static int PowerInt(int base, int exp);
};