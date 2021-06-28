#pragma once

#include "cudaUtility.h"

cudaError_t cudaWarpSum(uint16_t* first, uint16_t* second, uint16_t* output, uint32_t num);

cudaError_t cudaWarpSum(uint16_t* aDst, uint16_t* aSrc, uint32_t num);