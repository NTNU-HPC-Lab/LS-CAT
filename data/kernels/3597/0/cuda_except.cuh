#pragma once

#include "cuda_runtime.h"
#include <exception>
#include <string>
#include <sstream>

class CudaException : public std::exception
{
public:
	CudaException(cudaError_t error)
		: m_error(error)
		, m_message(cudaGetErrorString(m_error))
	{
	}

	const char* what() const override
	{
		return m_message.c_str();
	}

	cudaError_t getErrorCode() const
	{
		return m_error;
	}

private:
	cudaError_t m_error;
	std::string m_message;
};