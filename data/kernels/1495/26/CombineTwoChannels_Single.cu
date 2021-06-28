#include "includes.h"
__global__ static void CombineTwoChannels_Single(int* OCTData_2Channls, int* OCTData, int SizeX, int SizeY, int SizeZ)
{
// 這邊是 Denoise，把兩個 Channel 的資料相加
int id = blockIdx.y * gridDim.x * gridDim.z * blockDim.x +			// Y	=> Y * 250 * (2 * 1024)
blockIdx.x * gridDim.z * blockDim.x +							// X	=> X * (2 * 1024)
blockIdx.z * blockDim.x +										// Z	=> (Z1 * 1024 + Z2)
threadIdx.x;

// 這邊應該是不會發生，就當作例外判斷
if (id >= SizeX * SizeY * SizeZ)
{
printf("Combine Two Channel 有 Error!\n");
return;
}

int BoxSize = SizeX * SizeZ;										// 這邊沒有反掃，所以直接接上大小
int BoxIndex = id / BoxSize;
int BoxLeft = id % BoxSize;

OCTData[id] = (OCTData_2Channls[BoxIndex * 2 * BoxSize + BoxLeft] +
OCTData_2Channls[(BoxIndex * 2 + 1) * BoxSize + BoxLeft]) / 2;
}