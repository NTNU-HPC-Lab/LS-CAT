#pragma once

//#define NOMINMAX			// Use standard library min/max

#include <assert.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>


#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "GpuTimer.h"
#include "Utilities.h"
#include "Struct.h"


typedef unsigned int uint;
typedef unsigned char uchar;

////////////////////////////////////////////////////////////////////////////////
// GPU-specific common definitions
////////////////////////////////////////////////////////////////////////////////
#define HISTOGRAM_BIN_COUNT 256
#define UINT_BITS 32

#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

//May change on future hardware, so better parametrize the code
#define SHARED_MEMORY_BANKS 16

//Warps == subhistograms per threadblock
#define WARP_COUNT 6

//Threadblock size: must be a multiple of (4 * SHARED_MEMORY_BANKS)
//because of the bit permutation of threadIdx.x
#define HISTOGRAM_THREADBLOCK_SIZE (WARP_COUNT * WARP_SIZE)

//Shared memory per threadblock
#define HISTOGRAM_THREADBLOCK_MEMORY (WARP_COUNT * HISTOGRAM_BIN_COUNT)

#define UMUL(a, b) ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

extern "C" void initHistogram(void);
extern "C" void closeHistogram(void);

extern "C" void histogram(
	uint *d_Histogram,
	void *d_Data,
	uint byteCount);

extern "C" void RunAHEKernel(
	uchar* const			outputImage,					// Return value: rgba image 
	const uchar* const		originalImage,
	float* const			hue,
	float* const			saturation,
	uchar* const			value,
	uchar* const			valueBlurred,
	uchar* const			valueContrast,
	uchar* const			mask,
	const float* const		filterWeight,					// gaussian filter weights. The weights look like a bell shape.
	int						filterWidth,					// number of pixels in x and y directions for calculating average blurring
	int						rows,							// image size: number of rows
	int						cols							// image size: number of columns
	);
// Load libraries
//#pragma comment(lib, "cudart")