#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdint.h>
typedef short int16_t;
template<typename T>
inline T x265_clip3(T minVal, T maxVal, T a) { return x265_min(x265_max(minVal, a), maxVal); }
template<typename T>
inline T x265_min(T a, T b) { return a < b ? a : b; }
template<typename T>
inline T x265_max(T a, T b) { return a > b ? a : b; }
#define ALIGN_VAR_32(T, var) __declspec(align(32)) T var
enum FTYPE { DST, DCT4, DCT8, DCT16, DCT32, IDST, IDCT4, IDCT8, IDCT16, IDCT32 };
enum PTYPE { ASM , CPUBasic, CPU, GPUPlain, GPUMemShared, GPUMinMul, GPUOneStep, GPUAtomic };
enum FTYPESTAGE {DSTN, DSTT, DCT4N, DCT4T, DCT8N, DCT8T, DCT16N, DCT16T, DCT32N, DCT32T,
	IDSTN, IDSTT, IDCT4N, IDCT4T, IDCT8N, IDCT8T, IDCT16N, IDCT16T, IDCT32N, IDCT32T};