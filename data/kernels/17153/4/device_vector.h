#pragma once

#include "shape.cuh"
#include "exception.cuh"
#include <vector>
#include <memory>

class Tensor {
private:
	bool device_allocated;
	bool host_allocated;

	void allocateCudaMemory();
	void allocateHostMemory();

public:
	Shape shape;

	std::shared_ptr<float> data_device;
	std::shared_ptr<float> data_host;

	Tensor(size_t x_dim = 1, size_t y_dim = 1);
	Tensor(Shape shape);
	Tensor(Shape shape, std::vector<float>);

	void allocateMemory();
	void allocateMemoryIfNotAllocated(Shape shape);

	void copyHostToDevice();
	void copyDeviceToHost();

	float& operator[](const int index);
	const float& operator[](const int index) const;
};
