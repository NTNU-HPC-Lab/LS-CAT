/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  Wrapper functions for complex numbers which are not provided by 'cuComplex.h'
 */

#ifndef CUDA_COMPLEX_HPP
#define CUDA_COMPLEX_HPP

#ifdef __CUDACC__
 #define CUDA_CALLABLE_MEMBER __host__ __device__
#else
 #define CUDA_CALLABLE_MEMBER
#endif

// System includes
#include <cuComplex.h>
#include <complex>

/**
 *	Complex multiplication
 */
CUDA_CALLABLE_MEMBER static __inline__ cuFloatComplex operator*(cuFloatComplex lhs, 
                                                                cuFloatComplex rhs) 
{ 
	return cuCmulf(lhs, rhs); 
}

/**
 *	Complex addition
 */
CUDA_CALLABLE_MEMBER static __inline__ cuFloatComplex operator+(cuFloatComplex lhs, 
                                                                cuFloatComplex rhs) 
{ 
	return cuCaddf(lhs, rhs); 
}

/**
 *	Complex division
 */
CUDA_CALLABLE_MEMBER static __inline__ cuFloatComplex operator/(cuFloatComplex lhs, 
                                                                cuFloatComplex rhs) 
{ 
	return cuCdivf(lhs, rhs); 
}

/**
 *	Complex subtraction
 */
CUDA_CALLABLE_MEMBER static __inline__ cuFloatComplex operator-(cuFloatComplex lhs, 
                                                                cuFloatComplex rhs) 
{ 
	return cuCsubf(lhs, rhs); 
}

/**
 *	Complex norm
 */
CUDA_CALLABLE_MEMBER static __inline__ float cuCnormf(cuFloatComplex c) 
{ 
	const float creal = cuCrealf(c);
	const float cimag = cuCimagf(c);
	return creal*creal + cimag*cimag; 
}

/**
 *	Cast from std::complex<float> to cuFloatComplex
 */
static __inline__ cuFloatComplex make_cuFloatComplex(std::complex<float> c) 
{ 
	return make_cuFloatComplex(c.real(), c.imag()); 
}

/**
 *	Cast from std::complex<double> to cuFloatComplex
 */
static __inline__ cuFloatComplex make_cuFloatComplex(std::complex<double> c) 
{ 
	return make_cuFloatComplex(static_cast<float>(c.real()), 
	                           static_cast<float>(c.imag())); 
}

/**
 *	Cast from cuFloatComplex to std::complex<T>
 */
template< class T >
static __inline__ std::complex<T> make_stdComplex(cuFloatComplex c) 
{ 
	return std::complex<T>(static_cast<T>(cuCrealf(c)), 
	                       static_cast<T>(cuCimagf(c)));
}

/**
 *	Write to ostream
 */
template< class stream_t >
stream_t& operator<<(stream_t& os, cuFloatComplex c)
{
	os << "(" << cuCrealf(c) << "," << cuCimagf(c) << ")";
	return os;
}

#endif /* cudaComplex.hpp */
