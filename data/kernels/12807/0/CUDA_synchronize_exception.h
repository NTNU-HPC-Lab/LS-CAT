#include <iostream>
#include "CUDA_exception.h"

using namespace std;

/**
* Klasa pomocnicza do obslugi bledow, ktore moga wystapic w trakcie uzywania technologii CUDA
*/
class CUDA_synchronize_exception : public CUDA_exception
{
public:
	CUDA_synchronize_exception(){}
	virtual const char* what() const throw()
	{
		return "cudaDeviceSynchronize failed!";
	}
};