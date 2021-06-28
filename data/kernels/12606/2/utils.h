/* ==========================================================================	
   utils.h
   ==========================================================================
	
   Implementation of basic utils 

*/

#pragma once
#pragma warning(disable : 4995) 

#include <dynlink_d3d11.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <helper_cuda.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <string>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// COM-like Release macro
#ifndef SAFE_RELEASE
#define SAFE_RELEASE(a) if (a) { a->Release(); a = NULL; }
#endif

//Setup for making Debug.Log in Unity callable from here
void DebugInUnity(std::string message);

//Other helpers
void ProcessCudaError(std::string prefix);
void PrintTextureDesc(D3D11_TEXTURE2D_DESC desc);


