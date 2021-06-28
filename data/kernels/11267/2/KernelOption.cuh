#ifndef KERNELOPTION_CUH
#define KERNELOPTION_CUH

#include <math.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include <glm/vec3.hpp>
#include "CustomMath.cuh"

using namespace glm;

class KernelOption
{
public:
	__host__ __device__ KernelOption();
	__host__ __device__ KernelOption(float kernel);
	__host__ __device__ void setKernelSize(float kernel);
	__host__ __device__ void setPoly6Value();
	__host__ __device__ void setGradPoly6Value();
	__host__ __device__ void setSpikyKernelValue();
	__host__ __device__ void setViscoKernelValue();
	__host__ __device__ void setLaplacianPoly6Value();
	__host__ __device__ float getKernelSize();
	__host__ __device__ float getPoly6Value();
	__host__ __device__ float getGradPoly6Value();
	__host__ __device__ float getSpikyKernelValue();
	__host__ __device__ float getViscoKernelValue();
	__host__ __device__ float getLaplacianPoly6Value();
	__host__ __device__ vec3 getWPoly6Gradient(vec3 possDiff, float r2);
	__host__ __device__ float getWPoly6(float r2);
	__host__ __device__ float getWPoly6Laplacian(float r2);
	__host__ __device__ vec3 getWspikyGradient(vec3 possDiff, float r2);
	__host__ __device__ float getWviscosityLaplacian(float r2);
	__host__ __device__ float getKernel2();
private:
	float kernelSize;
	float kernel2;
	float poly6KernelValue;
	float gradPoly6KernelValue;
	float spikyKernelValue;
	float viscoKernelValue;
	float laplacianPoly6Value;
};

#endif