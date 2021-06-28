#pragma once

#include "tensor.cuh"
#include "utils.cuh"

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <algorithm>
#include <cfloat>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void operator_add(const Tensor *input1, const Tensor *input2,
				  Tensor *outputs);
void operator_add(const Tensor *input1, float value, Tensor *outputs);

void operator_sub(const Tensor *input1, const Tensor *input2,
				  Tensor *outputs);

void operator_mul(const Tensor *input1, const Tensor *input2,
				  Tensor *outputs);
void operator_mul(const Tensor *input1, float value, Tensor *outputs);

void operator_div(const Tensor *input1, const Tensor *input2,
				  Tensor *outputs);

void operator_log(const Tensor *input1, Tensor *outputs);

void operator_exp(const Tensor *input1, Tensor *outputs);

void operator_pow(const Tensor *input1, float e, Tensor *outputs);

void operator_matmul(const Tensor *input1, const Tensor *input2,
					 Tensor *outputs, int broadcast = 0);

void operator_transpose(const Tensor *input1, Tensor *outputs);

void operator_mean(const Tensor *input1, int dim, Tensor *outputs);

void operator_sum(const Tensor *input1, int dim, Tensor *outputs);