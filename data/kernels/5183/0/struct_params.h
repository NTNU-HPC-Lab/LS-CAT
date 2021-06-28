#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>

using namespace std;

/**
	use struct to 
	stores host and device params together
*/

struct Para
{
	float* h_a;
	float* h_b;
	float* h_c;

	float* d_a;
	float* d_b;
	float* d_c;
};


void struct_para_main();






