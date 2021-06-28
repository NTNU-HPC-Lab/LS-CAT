/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Wrapper class for N dimensional square matrices (N x N) aswell as 
 *  N-dimensional square matrices with 4-dimensional vector as elements. 
 *  The matrix is stored in ROW MAJOR order, which means the LAST index is 
 *  varying the most. By default the memory is aligned on 64 byte which can
 *  be disable with the macro MATRIX_NO_ALIGN.
 *	 
 *  The implementation will depend on the macro's MATRIX_USE_STL and 
 *  MATRIX_USE_CARRAY. If MATRIX_USE_CARRAY is defined plain c-arrays are used 
 *  as a container, otherwise std::vector is used i.e MATRIX_USE_STL is defined 
 *  by default.
 *
 *  [EXAMPLE]
 *  Creating a simple 2x2 matrix
 *  
 *  	a	b
 *  	c	d
 *
 *  Initialization :
 *
 *  matND<double> matrix(2);	
 *	
 *  to use default initialization (i.e fill everything with 0.0)
 *
 *  matND<double> matrix(2,0.0)
 *
 *  To access the elements e.g get the value of c :
 *
 *  double c = matrix(1,0);
 *
 *  or
 *
 *  double c = matrix[2];
 *
 *  which are both equivalent.
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

// System includes
#include <cstddef>
#include <algorithm>
#ifdef _WIN32
 #include <malloc.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
 #define MATRIX_FORCE_INLINE
 #ifndef MATRIX_NO_ALIGN
  #define MATRIX_FORCE_ALIGNED	__declspec(align(64))
 #else
  #define MATRIX_FORCE_ALIGNED
 #endif
#else
 #define MATRIX_FORCE_INLINE	__attribute__((always_inline)) 
 #ifndef MATRIX_NO_ALIGN
  #define MATRIX_FORCE_ALIGNED	__attribute__((aligned(64)))
 #else
  #define MATRIX_FORCE_ALIGNED
 #endif
#endif

#if __cplusplus >= 201103L
 #define NOEXCEPT noexcept
#else
 #define NOEXCEPT
#endif

// MATRIX_USE_STL and MATRIX_ALIGN is default
#ifndef MATRIX_USE_CARRAY
 #undef  MATRIX_USE_STL
 #define MATRIX_USE_STL
 #ifndef MATRIX_NO_ALIGN
  #define VEC_T(ALIGNMENT) std::vector<T, aligned_allocator<T, (ALIGNMENT) > >
 #else
  #define VEC_T(ALIGNMENT) std::vector<T>
 #endif
#endif


template< typename T, unsigned int alignment >
class aligned_allocator
{
public:
	// === typedefs ===
	typedef T*              pointer;
	typedef T const*        const_pointer;
	typedef T&              reference;
	typedef T const&        const_reference;
	typedef T               value_type;
	typedef std::size_t     size_type;
	typedef std::ptrdiff_t  difference_type;

	template< typename U >
	struct rebind 
	{
		typedef aligned_allocator< U, alignment > other;
	};

	/**
	 *	Constructor - default
	 */
	aligned_allocator() NOEXCEPT 
	{}

	/**
	 *	Constructor - copy
	 *	@param 	a 	aligned_allocator
	 */
	aligned_allocator(aligned_allocator const& a) NOEXCEPT 
	{}


	/**
	 *	Constructor - copy
	 *	@param 	a 	aligned_allocator
	 */
	template< typename U >
	aligned_allocator(aligned_allocator<U, alignment> const& a) NOEXCEPT 
	{}

	/**
	 *	Allocate aligned memory
	 *	@param 	size 	allocate size bytes
	 *	@return pointer to address of the memory location
	 */
	pointer allocate(size_type size) 
	{
		pointer p;

#ifdef _WIN32
		p = reinterpret_cast<pointer>(_aligned_malloc(size*sizeof(T), alignment));
		if(p == NULL)
			throw std::bad_alloc();
#else // UNIX
		if(posix_memalign(reinterpret_cast<void**>(&p), alignment, size*sizeof(T)))
			throw std::bad_alloc();
#endif
		return p;
	}

	/**
	 *	Deallocate aligned memory
	 *	@param  p		address of the beginning of the memory location
	 *	@param 	size 	deallocate size bytes
	 */
	void deallocate(pointer p, size_type n) NOEXCEPT
	{
#ifdef _WIN32
		_aligned_free(p);
#else
		std::free(p);
#endif 
	}

	/**
	 *	Returns the maximum theoretically possible value of n, for which the 
	 *	call to std::allocator<T>::allocate(n,0) could succeed.
	 *	@return the maximum supported allocation size
	 */
	size_type max_size() const NOEXCEPT
	{
		std::allocator<T> a;
		return a.max_size();
	}

	/**
	 *	Constructs an object of type T in allocated uninitialized storage 
	 *	pointed to by p, using placement-new
	 *	@param 	p		pointer to allocated uninitialized storage
	 *	@param 	t		the value to use as the copy constructor argument
	 *	@param 	args...	the constructor arguments to use
	 */
#if __cplusplus >= 201103L 
	template <typename U, class... Args>
	void construct(U* p, Args&&... args) 
	{
		new ((void*)p) U(std::forward<Args>(args)...);
	}
#else
	void construct(pointer p, const_reference t) 
	{
		new((void *)p) T(t);
	}
#endif

	/**
	 *	Returns the actual address of x
	 *	@param	x	the object to acquire address of
	 */
	pointer address(reference x ) const
	{
		return &x;
	}

	const_pointer address(const_reference x ) const
	{
		return &x;
	}

	/**
	 *	Destructs an object in allocated storage 
	 *	@param 	p	pointer to the object that is going to be destroyed
	 */
	template< typename U >
	void destroy(U* p) 
	{
		p->~U();
	}

	// === Operator ===
	bool operator==(aligned_allocator const& a2) const NOEXCEPT 
	{
		return true;
	}

	bool operator!=(aligned_allocator const& a2) const NOEXCEPT 
	{
		return false;
	}

	template <typename U, unsigned int U_alignment>
	bool operator==(aligned_allocator<U, U_alignment> const& b) const NOEXCEPT 
	{
		return false;
	}

	template <typename U, unsigned int U_alignment>
	bool operator!=(aligned_allocator<U, U_alignment> const& b) const NOEXCEPT 
	{
		return true;
	}

}; 



#if defined( MATRIX_USE_STL )
#include <vector>

template < typename T >
class matND : private VEC_T(64)
{
public:
	// === Constructor ===
	
	/**
	 *	Constructor
	 *	@param	N		one dimension of the matrix (N x N)
	 *	@param	val		initial value (default : 0)
	 */
	matND(std::size_t N, T val = T(0))
		: VEC_T(64)(N*N,val), N_(N)
	{}

	/**
	 *	Constructor
	 *	@param	N		one dimension of the matrix (N x N)
	 *	@param	begin	pointer to the begin of an array of size N*N
	 *	@param	end		pointer to the end (past the end)
	 */
	matND(std::size_t N, const T* begin, const T* end)
		: VEC_T(64)(begin, end), N_(N)
	{}

	/**
	 *	Copy-Constructor
	 *	@param	v		matND used for copy constructing
	 */
	matND(const matND& v)
		: std::vector<T>(v.N()*v.N()), N_(v.N())
	{
		std::copy(v.begin(), v.end(), VEC_T(64)::begin());
	}
	
	// === Access ===
	
	/**
	 *	Access (const reference)	
	 *	@param	i		index in array [0, N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline T const& operator[](std::size_t i) const
	{
		return VEC_T(64)::operator[](i);
	}

	/**
	 *	Access (reference)	
	 *	@param	i		index in array [0, N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline T& operator[](std::size_t i) 
	{
		return VEC_T(64)::operator[](i);
	}
	
	/**
	 *	Access (const reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 */
	MATRIX_FORCE_INLINE 
	inline T const& operator()(std::size_t i, std::size_t j) const
	{
		return VEC_T(64)::operator[](i*N_ + j);
	}

	/**
	 *	Access (reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 */
	MATRIX_FORCE_INLINE
	inline T& operator()(std::size_t i, std::size_t j)
	{
		return VEC_T(64)::operator[](i*N_ + j);
	}
		
	// === Data access ===
	
	/**
	 *	Return one dimension of the matrix
	 *	@return N
	 */
	inline std::size_t N() const 
	{ 
		return N_; 
	}

	/**
	 *	Size of the underlying array
	 *	@return N*N
	 */
	inline std::size_t size() const 
	{ 
		return N_*N_; 
	}

	/**
	 *	Pointer to the first element (const pointer)
	 */
	inline const T* data() const 
	{
		return &VEC_T(64)::operator[](0); 
	}

	/**
	 *	Pointer to the first element (pointer)
	 */
	inline T* data() 
	{ 
		return &VEC_T(64)::operator[](0); 
	} 
	
private:
	std::size_t N_;
};

#else

template < class T >
class matND
{
public:
	
	// === Constructor ===
	
	/**
	 *	Constructor
	 *	@param	N		one dimension of the matrix (N x N)
	 *	@param	val		initial value (default : 0)
	 */
	matND(std::size_t N, T val = T(0))
		: value_(new T[N*N]), N_(N)
	{
		std::fill(value_, value_+N*N, val);
	}

	/**
	 *	Constructor
	 *	@param	N		one dimension of the matrix (N x N)
	 *	@param	begin	pointer to the begin of an array of size N*N
	 *	@param	end		pointer to the end (past the end)
	 */
	matND(std::size_t N, const T* begin, const T* end)
		: value_(new T[N*N]), N_(N)
	{
		std::copy(begin, end, value_);
	}
	
	/**
	 *	Copy-Constructor
	 *	@param	v		matND used for copy constructing
	 */
	matND(const matND& v)
		: value_(new T[v.N()*v.N()]), N_(v.N())
	{
		std::copy(v.data(), v.data()+v.N()*v.N(), value_);
	}
	
	// === Destructor ===
	
	~matND() { delete value_; }
	
	// === Access ===
	
	/**
	 *	Access (const reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 */
	MATRIX_FORCE_INLINE
	inline T const& operator()(std::size_t i, std::size_t j) const
	{
		return value_[i*N_ + j];
	}
	
	/**
	 *	Access (reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 */
	MATRIX_FORCE_INLINE 
	inline T& operator()(std::size_t i, std::size_t j)
	{
		return value_[i*N_ + j];
	}
	
	/**
	 *	Access (const reference)	
	 *	@param	i		index in array [0, N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline T const& operator[](std::size_t i) const
	{
		return value_[i];
	}

	/**
	 *	Access (reference)	
	 *	@param	i		index in array [0, N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline T& operator[](std::size_t i) 
	{
		return value_[i];
	}

	// === Data access ===
	
	/**
	 *	Pointer to the first element (const pointer)
	 */
	inline const T* data() const 
	{ 
		return value_; 
	}

	/**
	 *	Pointer to the first element (pointer)
	 */
	inline T* data() 
	{ 
		return value_; 
	} 
	
	/**
	 *	Return one dimension of the vector
	 *	@return N
	 */
	inline std::size_t N() const 
	{ 
		return N_; 
	}
	
	/**
	 *	Size of the underlying array
	 *	@return N*N
	 */
	inline std::size_t size() const 
	{ 
		return N_*N_; 
	}

private:
	MATRIX_FORCE_ALIGNED T* value_;
	std::size_t N_;
};

#endif /* MATRIX_USE_STL */


#if defined( MATRIX_USE_STL )

template < class T >
class matN4D : private VEC_T(64)
{
public:

	// === Constructor ===
	
	/**
	 *	Constructor
	 *	@param	N		one dimension of the matrix (N x N x 4)
	 *	@param	val		initial value (default : 0)
	 */
	matN4D(std::size_t N, T val = T(0))
		: VEC_T(64)(N*N*4,val), N_(N)
	{}

	/**
	 *	Constructor
	 *	@param	N		one dimension of the vector (N x N x 4)
	 *	@param	begin	pointer to the begin of an array of size 4*N*N
	 *	@param	end		pointer to the end (past the end)
	 */
	matN4D(std::size_t N, const T* begin, const T* end)
		: VEC_T(64)(begin, end), N_(N)
	{}
	
	// === Access ===
	
	/**
	 *	Access (const reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 *	@param  k		vector index [0,3]
	 */
	MATRIX_FORCE_INLINE 
	inline T const& operator()(std::size_t i, std::size_t j, std::size_t k) const
	{
		return VEC_T(64)::operator[](4*(N_*i + j) + k);
	}

	/**
	 *	Access (reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 *	@param  k		vector index [0,3]
	 */
	MATRIX_FORCE_INLINE
	inline T& operator()(std::size_t i, std::size_t j, std::size_t k)
	{
		return VEC_T(64)::operator[](4*(N_*i + j) + k);
	}
	
	/**
	 *	Access (const reference)	
	 *	@param	i		index in array [0, 4*N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline T const& operator[](std::size_t i) const
	{
		return VEC_T(64)::operator[](i);
	}

	/**
	 *	Access (reference)	
	 *	@param	i		index in array [0, 4*N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline T& operator[](std::size_t i) 
	{
		return VEC_T(64)::operator[](i);
	}
	
		
	// === Data access ===
	
	/**
	 *	Return one dimension of the vector
	 *	@return N
	 */
	inline std::size_t N() const 
	{ 
		return N_; 
	}

	/**
	 *	Size of the underlying array
	 *	@return N*N*4
	 */
	inline std::size_t size() const 
	{ 
		return N_*N_*4; 
	}

	/**
	 *	Pointer to the first element (const pointer)
	 */
	inline const T* data() const 
	{
		return &VEC_T(64)::operator[](0); 
	}

	/**
	 *	Pointer to the first element (pointer)
	 */
	inline T* data() 
	{ 
		return &VEC_T(64)::operator[](0); 
	} 

private:
	std::size_t N_;
};

#else

template < class T >
class matN4D
{
public:
	
	// === Constructor ===
	
	/**
	 *	Constructor
	 *	@param	N		one dimension of the matrix (N x N x 4)
	 *	@param	val		initial value (default : 0)
	 */
	matN4D(std::size_t N, T val = T(0))
		: value_(new T[N*N*4]), N_(N)
	{
		std::fill(value_, value_+N*N*4, val);
	}

	/**
	 *	Constructor
	 *	@param	N		one dimension of the vector (N x N x 4)
	 *	@param	begin	pointer to the begin of an array of size 4*N*N
	 *	@param	end		pointer to the end (past the end)
	 */
	matN4D(std::size_t N, const T* begin, const T* end)
		: value_(new T[N*N*4]), N_(N)
	{
		// Visual Studio has a diffrent version of std::copy (from xutility)
		// we want to avoid using that
#ifdef _MSC_VER
		const std::size_t NN_const = N*N*4;
		for(std::size_t i = 0; i < NN_const; ++i)
			value_[i] = *(begin+i);
#else
		std::copy(begin, end, value_);
#endif
	}
	
	// === Destructor ===
	
	~matN4D() { delete value_; }
	
	// === Access ===
	
	/**
	 *	Access (const reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 *	@param  k		vector index [0,3]
	 */
	MATRIX_FORCE_INLINE 
	inline T const& operator()(std::size_t i, std::size_t j, std::size_t k) const
	{
		return value_[4*(N_*i + j) + k];
	}

	/**
	 *	Access (reference)	
	 *	@param	i		row index in [0, N)
	 *	@param	j		column index in [0, N)
	 *	@param  k		vector index [0,3]
	 */
	MATRIX_FORCE_INLINE
	inline T& operator()(std::size_t i, std::size_t j, std::size_t k)
	{
		return value_[4*(N_*i + j) + k];
	}
	
	/**
	 *	Access (const reference)	
	 *	@param	i		index in array [0, 4*N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline T const& operator[](std::size_t i) const
	{
		return value_[i];
	}

	/**
	 *	Access (reference)	
	 *	@param	i		index in array [0, 4*N*N)
	 */
	MATRIX_FORCE_INLINE 
	inline T& operator[](std::size_t i) 
	{
		return value_[i];
	}
	
	// === Data access ===

	/**
	 *	Return one dimension of the vector
	 *	@return N
	 */
	inline std::size_t N() const 
	{ 
		return N_; 
	}

	/**
	 *	Size of the underlying array
	 *	@return N*N*4
	 */
	inline std::size_t size() const 
	{ 
		return N_*N_*4; 
	}

	/**
	 *	Pointer to the first element (const pointer)
	 */
	inline const T* data() const 
	{ 
		return value_; 
	}

	/**
	 *	Pointer to the first element (pointer)
	 */
	inline T* data() 
	{ 
		return value_; 
	} 

private:
	MATRIX_FORCE_ALIGNED T* value_;
	std::size_t N_;
};

#endif /* MATRIX_USE_STL */


#undef MATRIX_FORCE_INLINE 
#undef MATRIX_FORCE_ALIGNED
#undef NOEXCEPT

#endif /* matrix.hpp */
