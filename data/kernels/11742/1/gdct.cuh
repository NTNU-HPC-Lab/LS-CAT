#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "x86dct.h"
enum Type { D1, D2, ID1, ID2 };
__device__ const int16_t ta[4][4] =
{
	{ 29, 55, 74, 84 },
	{ 74, 74, 0, -74 },
	{ 84, -29, -74, 55 },
	{ 55, -84, 74, -29 }
};

__device__ const int16_t t4[4][4] =
{
	{ 64, 64, 64, 64 },
	{ 83, 36, -36, -83 },
	{ 64, -64, -64, 64 },
	{ 36, -83, 83, -36 }
};

__device__ const int16_t t8[8][8] =
{
	{ 64, 64, 64, 64, 64, 64, 64, 64 },
	{ 89, 75, 50, 18, -18, -50, -75, -89 },
	{ 83, 36, -36, -83, -83, -36, 36, 83 },
	{ 75, -18, -89, -50, 50, 89, 18, -75 },
	{ 64, -64, -64, 64, 64, -64, -64, 64 },
	{ 50, -89, 18, 75, -75, -18, 89, -50 },
	{ 36, -83, 83, -36, -36, 83, -83, 36 },
	{ 18, -50, 75, -89, 89, -75, 50, -18 }
};

__device__ const int16_t t16[16][16] =
{
	{ 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64 },
	{ 90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90 },
	{ 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89 },
	{ 87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87 },
	{ 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83 },
	{ 80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80 },
	{ 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75 },
	{ 70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70 },
	{ 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64 },
	{ 57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57 },
	{ 50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50 },
	{ 43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43 },
	{ 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36 },
	{ 25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25 },
	{ 18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18 },
	{ 9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9 }
};

__device__ const int16_t t32[32][32] =
{
	{ 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64 },
	{ 90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4, -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90 },
	{ 90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -25, -9, 9, 25, 43, 57, 70, 80, 87, 90 },
	{ 90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13, 13, 38, 61, 78, 88, 90, 85, 73, 54, 31, 4, -22, -46, -67, -82, -90 },
	{ 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89 },
	{ 88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22, -22, -61, -85, -90, -73, -38, 4, 46, 78, 90, 82, 54, 13, -31, -67, -88 },
	{ 87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80, 90, 70, 25, -25, -70, -90, -80, -43, 9, 57, 87 },
	{ 85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31, 31, 78, 90, 61, 4, -54, -88, -82, -38, 22, 73, 90, 67, 13, -46, -85 },
	{ 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83 },
	{ 82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38, -38, -88, -73, -4, 67, 90, 46, -31, -85, -78, -13, 61, 90, 54, -22, -82 },
	{ 80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80, -80, -9, 70, 87, 25, -57, -90, -43, 43, 90, 57, -25, -87, -70, 9, 80 },
	{ 78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46, 46, 90, 38, -54, -90, -31, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78 },
	{ 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75 },
	{ 73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54, -54, -85, 4, 88, 46, -61, -82, 13, 90, 38, -67, -78, 22, 90, 31, -73 },
	{ 70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90, -25, 80, 57, -57, -80, 25, 90, 9, -87, -43, 70 },
	{ 67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61, 61, 73, -46, -82, 31, 88, -13, -90, -4, 90, 22, -85, -38, 78, 54, -67 },
	{ 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64 },
	{ 61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67, -67, -54, 78, 38, -85, -22, 90, 4, -90, 13, 88, -31, -82, 46, 73, -61 },
	{ 57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57, -57, 80, 25, -90, 9, 87, -43, -70, 70, 43, -87, -9, 90, -25, -80, 57 },
	{ 54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73, 73, 31, -90, 22, 78, -67, -38, 90, -13, -82, 61, 46, -88, 4, 85, -54 },
	{ 50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50, 50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50 },
	{ 46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78, -78, -4, 82, -73, -13, 85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46 },
	{ 43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43, -43, 90, -57, -25, 87, -70, -9, 80, -80, 9, 70, -87, 25, 57, -90, 43 },
	{ 38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82, 82, -22, -54, 90, -61, -13, 78, -85, 31, 46, -90, 67, 4, -73, 88, -38 },
	{ 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36 },
	{ 31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85, -85, 46, 13, -67, 90, -73, 22, 38, -82, 88, -54, -4, 61, -90, 78, -31 },
	{ 25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25, -25, 70, -90, 80, -43, -9, 57, -87, 87, -57, 9, 43, -80, 90, -70, 25 },
	{ 22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88, 88, -67, 31, 13, -54, 82, -90, 78, -46, 4, 38, -73, 90, -85, 61, -22 },
	{ 18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18, 18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18 },
	{ 13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90, -90, 82, -67, 46, -22, -4, 31, -54, 73, -85, 90, -88, 78, -61, 38, -13 },
	{ 9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9, -9, 25, -43, 57, -70, 80, -87, 90, -90, 87, -80, 70, -57, 43, -25, 9 },
	{ 4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90, 90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4 }
};
class Gpu
{
public:
	Gpu(int n);
	~Gpu();

	int GetAge() const;        // accessor function
	void SetAge(int age);      // accessor function
	void Meow();
private:                      // begin private section
	int itsAge;                // member variable
	char * string;
};
template<FTYPE type>
void gpuLessMulTransform(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);

template void gpuLessMulTransform<DST>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuLessMulTransform<DCT4>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuLessMulTransform<DCT8>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuLessMulTransform<DCT16>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuLessMulTransform<DCT32>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuLessMulTransform<IDST>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuLessMulTransform<IDCT4>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuLessMulTransform<IDCT8>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuLessMulTransform<IDCT16>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuLessMulTransform<IDCT32>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);

template<FTYPE type>
void gpuTransformPlain(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);

template void gpuTransformPlain<DST>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformPlain<DCT4>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformPlain<DCT8>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformPlain<DCT16>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformPlain<DCT32>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformPlain<IDST>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformPlain<IDCT4>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformPlain<IDCT8>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformPlain<IDCT16>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformPlain<IDCT32>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);

template<FTYPE type>
void gpuTransformShared(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);

template void gpuTransformShared<DST>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformShared<DCT4>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformShared<DCT8>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformShared<DCT16>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformShared<DCT32>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformShared<IDST>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformShared<IDCT4>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformShared<IDCT8>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformShared<IDCT16>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransformShared<IDCT32>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);

template<FTYPE type>
void gpuTransform1Step(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);

template void gpuTransform1Step<DST>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransform1Step<DCT4>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransform1Step<DCT8>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransform1Step<DCT16>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransform1Step<DCT32>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransform1Step<IDST>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransform1Step<IDCT4>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransform1Step<IDCT8>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransform1Step<IDCT16>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);
template void gpuTransform1Step<IDCT32>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n);

template<FTYPE ftype>
void gpuTransform1StepBatch(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n, int m);

template void gpuTransform1StepBatch<DST>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n, int m);
template void gpuTransform1StepBatch<DCT4>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n, int m);
template void gpuTransform1StepBatch<DCT8>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n, int m);
template void gpuTransform1StepBatch<DCT16>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n, int m);
template void gpuTransform1StepBatch<DCT32>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n, int m);
template void gpuTransform1StepBatch<IDST>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n, int m);
template void gpuTransform1StepBatch<IDCT4>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n, int m);
template void gpuTransform1StepBatch<IDCT8>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n, int m);
template void gpuTransform1StepBatch<IDCT16>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n, int m);
template void gpuTransform1StepBatch<IDCT32>(const int16_t* h_src, int16_t* h_dst, int shift1, int shift2, int n, int m);

void cudaAlloc(int n);
void cudaAlloc(int n, int m);
void cudaDestroy();