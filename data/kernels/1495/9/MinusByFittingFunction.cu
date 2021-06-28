#include "includes.h"
__global__ static void MinusByFittingFunction(int* OCTData, float* PolyValue, int SizeZ)
{
// 這邊要減掉 Fitting Data
int id = blockIdx.y * gridDim.x * gridDim.z * blockDim.x +			// Y	=> Y * 250 * (2 * 1024)
blockIdx.x * gridDim.z * blockDim.x +							// X	=> X * (2 * 1024)
blockIdx.z * blockDim.x +										// Z	=> (Z1 * 1024 + Z2)
threadIdx.x;

// 先拿出他是第幾個 Z
int idZ = id % SizeZ;

// 減掉預測的值
OCTData[id] -= PolyValue[idZ];
}