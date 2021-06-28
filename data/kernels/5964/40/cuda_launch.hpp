/*
 * cuda_launch.hpp
 *
 *  Created on: Jan 14, 2019
 *      Author: i-bird
 */

#ifndef CUDA_LAUNCH_HPP_
#define CUDA_LAUNCH_HPP_

#include "cuda_kernel_error_checker.hpp"

#ifdef CUDA_GPU

#if defined(SE_CLASS1) || defined(CUDA_CHECK_LAUNCH)

template<typename lambda_funct_type, typename ... Args_type>
__global__ void lambda_launcher(lambda_funct_type lbf, Args_type ... args)
{
	lbf(args...);
}

#define CUDA_LAUNCH(cuda_call,ite, ...) \
        {\
		cudaDeviceSynchronize(); \
		{\
			cudaError_t e = cudaGetLastError();\
			if (e != cudaSuccess)\
			{\
				std::string error = cudaGetErrorString(e);\
				std::cout << "Cuda an error has occurred before this CUDA_LAUNCH, detected in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
			}\
		}\
	    CHECK_SE_CLASS1_PRE\
	    if (ite.wthr.x != 0)\
		{cuda_call<<<ite.wthr,ite.thr>>>(__VA_ARGS__);}\
		cudaDeviceSynchronize(); \
		{\
			cudaError_t e = cudaGetLastError();\
			if (e != cudaSuccess)\
			{\
				std::string error = cudaGetErrorString(e);\
				std::cout << "Cuda Error in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
			}\
			CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
		}\
        }

#define CUDA_LAUNCH_DIM3(cuda_call,wthr,thr, ...) \
        {\
		cudaDeviceSynchronize(); \
		{\
			cudaError_t e = cudaGetLastError();\
			if (e != cudaSuccess)\
			{\
				std::string error = cudaGetErrorString(e);\
				std::cout << "Cuda an error has occurred before this CUDA_LAUNCH, detected in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
			}\
		}\
	    CHECK_SE_CLASS1_PRE\
		cuda_call<<<wthr,thr>>>(__VA_ARGS__);\
		cudaDeviceSynchronize(); \
		{\
			cudaError_t e = cudaGetLastError();\
			if (e != cudaSuccess)\
			{\
				std::string error = cudaGetErrorString(e);\
				std::cout << "Cuda Error in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
			}\
			CHECK_SE_CLASS1_POST(#cuda_call,__VA_ARGS__)\
		}\
        }

#define CUDA_CHECK() \
        {\
		cudaDeviceSynchronize(); \
		{\
			cudaError_t e = cudaGetLastError();\
			if (e != cudaSuccess)\
			{\
				std::string error = cudaGetErrorString(e);\
				std::cout << "Cuda an error has occurred before, detected in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
			}\
		}\
	    CHECK_SE_CLASS1_PRE\
		cudaDeviceSynchronize(); \
		{\
			cudaError_t e = cudaGetLastError();\
			if (e != cudaSuccess)\
			{\
				std::string error = cudaGetErrorString(e);\
				std::cout << "Cuda Error in: " << __FILE__ << ":" << __LINE__ << " " << error << std::endl;\
			}\
			CHECK_SE_CLASS1_POST("no call","no args")\
		}\
        }

#else

#define CUDA_LAUNCH(cuda_call,ite, ...) \
	    if (ite.wthr.x != 0)\
		{cuda_call<<<ite.wthr,ite.thr>>>(__VA_ARGS__);}

#define CUDA_LAUNCH_DIM3(cuda_call,wthr,thr, ...) \
		cuda_call<<<wthr,thr>>>(__VA_ARGS__);

#define CUDA_CHECK()

#endif

#endif

#endif /* CUDA_LAUNCH_HPP_ */
