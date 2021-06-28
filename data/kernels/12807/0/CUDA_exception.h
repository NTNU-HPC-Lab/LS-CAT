#ifndef CUDA_EXCEPTION_H
#define CUDA_EXCEPTION_H

#include <iostream>
#include <exception>

using namespace std;

/**
* Klasa pomocnicza do obslugi bledow, ktore moga wystapic w trakcie uzywania technologii CUDA
*/
class CUDA_exception : public exception
{
public:
	CUDA_exception(){}
	virtual const char* what() const throw() = 0;
};

#endif