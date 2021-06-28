// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#include <Windows.h>
#include <cstdio>
#include <cstdlib>
#include <tchar.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
using namespace std;


// TODO: reference additional headers your program requires here

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "device_functions.h"
#include "cuda_device_runtime_api.h"

#ifdef __INTELLISENSE__
	void __syncthreads();
#endif
#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#define imin(a, b) (a<b ? a : b)


#define N               33 * 1024
#define ThreadsPerBlock 256
#define BlocksPerGrid   imin(32, (N+ThreadsPerBlock -1)/ThreadsPerBlock)
