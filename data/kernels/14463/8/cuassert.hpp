/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Function to handle CUDA runtime errors.
 *  The function follows the same API as the standard cassert.
 *	
 *  cuassert( cudaMalloc(&device_array, 1024) );
 */

#ifndef CUASSERT_HPP
#define CUASSERT_HPP

// System includes
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

// Local includes
#ifndef NO_COLOR_TERMINAL
 #include "ConsoleColor.hpp"
#endif

// Do nothing if NDEBUG is defined
#ifndef NDEBUG
 #define cuassert(ans) { _cuassert((ans), __FILE__, __LINE__); }
 #define CUDA_CHECK_KERNEL  \
         { _cuassert(cudaPeekAtLastError(), __FILE__, __LINE__ - 1); }
#else
 #define cuassert(ans) { (ans); }
 #define CUDA_CHECK_KERNEL
#endif

#ifndef NO_COLOR_TERMINAL

static inline void _cuassert(cudaError_t code, const char *file, int line)
{
	if(code != cudaSuccess) 
	{
		ConsoleColor cc;
		cc.set_color(ConsoleColor::COLOR_WHITE);
		std::cerr << file << ":" << line;
		cc.set_color(ConsoleColor::COLOR_RED);
		std::cerr << " CudaError: ";
		cc.reset_color();
		std::cerr << cudaGetErrorString(code) << std::endl;
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

#else

static inline void _cuassert(cudaError_t code, const char *file, int line)
{
	if(code != cudaSuccess) 
	{
		std::cerr << file << ":" << line;
		std::cerr << " CudaError: " << cudaGetErrorString(code) << std::endl;
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

#endif /* NO_COLOR_TERMINAL */

#endif /* cuassert.hpp */
