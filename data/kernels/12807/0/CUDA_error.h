#include <iostream>
#include "CUDA_exception.h"

using namespace std;

/**
* Klasa pomocnicza do obslugi bledow, ktore moga wystapic w trakcie uzywania technologii CUDA
*/
class CUDA_error : public CUDA_exception
{
private:
	cudaError_t code;
public:
	CUDA_error(){}
	CUDA_error(cudaError_t c) : code(c)	{}
	virtual const char* what() const throw()
	{
		return cudaGetErrorString(code);
	}
};