#ifndef QUERY_CUH
#define QUERY_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdio.h>
#include <math.h>

class Query{
public:
	__host__ __device__ Query(){
	};
	__host__ __device__ Query(int _min, int _max){
		min = _min;
		max = _max;
	};
	int min;
	int max;
};

#endif