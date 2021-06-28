#include <iostream>
#include "CUDA_exception.h"

using namespace std;

/**
* Klasa pomocnicza do obslugi bledow, ktore moga wystapic w trakcie uzywania technologii CUDA
*/
class CUDA_kernel_exception : public CUDA_exception
{
public:
	CUDA_kernel_exception(){}
	virtual const char* what() const throw()
	{
		return "Kernel launch failed!";
	}
};