#include "includes.h"
__global__ static void ReverseBackScanData(int* OCTData, int SizeX, int SizeY, int SizeZ)
{
// 這邊是要反轉 反掃的資料
int id = (blockIdx.y * 2 + 1) * gridDim.x * 2 * gridDim.z * blockDim.x +			// Y	=> (Y * 2 + 1) * (2 * 1024)						=> 1, 3, 5, 7, 9
blockIdx.x * gridDim.z * blockDim.x +											// X	=> X * (125 * 2) * (2 * 1024)
blockIdx.z * blockDim.x +														// Z	=> (Z1 * 1024 + Z2)
threadIdx.x;

int changeID = (blockIdx.y * 2 + 1) * gridDim.x * 2 * gridDim.z * blockDim.x +		// Y	=> (Y * 2 + 1) * (2 * 1024)						=> 1, 3, 5, 7, 9
(gridDim.y * 2 - blockIdx.x - 1) * gridDim.z * blockDim.x +						// X	=> (250 - X - 1) * (125 * 2) * (2 * 1024)
blockIdx.z * blockDim.x +														// Z	=> (Z1 * 1024 + Z2)
threadIdx.x;

int value = OCTData[id];
OCTData[id] = OCTData[changeID];
OCTData[changeID] = value;
}