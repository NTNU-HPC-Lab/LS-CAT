#pragma once
#include "tensor.cuh"

class CCELoss {
public:
	float cost(Tensor predictions, Tensor target);
	Tensor dCost(Tensor predictions, Tensor target, Tensor dY);
};
