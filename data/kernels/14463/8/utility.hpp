/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Several useful utility functions and classes like Timer and Progressbar
 */

#ifndef UTILITY_HPP
#define UTILITY_HPP

// System includes 
#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "error.hpp"

#ifdef _WIN32
 #define NOMINMAX
 #include <windows.h>
 #undef NOMINMAX
#else
 #include <unistd.h>
 #include <sys/time.h>
#endif

// Compiler Hints
#ifdef _WIN32
 #define FORCE_INLINE
 #define NO_RETURN     __declspec(noreturn)
#else
 #define FORCE_INLINE  __attribute__((always_inline)) 
 #define NO_RETURN     __attribute__((noreturn))
#endif

// Compile-time warnings #pragma message WARN("message")
#if _MSC_VER
 #define STRINGIFICATION_IMPL(x) #x
 #define STRINGIFICATION(x) STRINGIFICATION_IMPL(x)
 #define FILE_LINE_LINK __FILE__ "(" STRINGIFICATION(__LINE__) ") : "
 #define WARN(exp) (FILE_LINE_LINK "WARNING: " exp)
#else
 #define WARN(exp) ("WARNING: " exp)
#endif

// Branch prediction hints
#if defined(__GNUC__) && __GNUC__ >= 4
 #define LIKELY(x)   (__builtin_expect((x), 1))
 #define UNLIKELY(x) (__builtin_expect((x), 0))
#else
 #define LIKELY(x)   (x)
 #define UNLIKELY(x) (x)
#endif

// Windows workaround
#ifdef SPRINTF
#undef SPRINTF
#endif
#ifdef _MSC_VER
 #define SPRINTF(buf, ...) sprintf_s((buf), __VA_ARGS__)
#else
 #define SPRINTF(buf, ...) std::sprintf((buf), __VA_ARGS__)
#endif

#if __cplusplus >= 201103L || _MSC_VER >= 1700
#include <chrono>

class Timer
{
public:
	Timer()
		:	start_was_called_(false)
	{}
	
	/**
	 *	Start the timer
	 */
	inline void start()
	{
		t_start_ = std::chrono::high_resolution_clock::now();
		start_was_called_ = true;
	}
	
	/**
	 *	Stop the timer and update total time
	 *	@return	the time in seconds since start()
	 */
	inline double stop()
	{
		if(!start_was_called_) 
			WARNING("calling stop() without previously calling start()");

		t_end_ = std::chrono::high_resolution_clock::now();
		start_was_called_ = false;
		return std::chrono::duration<double>(t_end_ - t_start_).count();
	}
	
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> t_start_, t_end_;
	bool start_was_called_;
};

#else

#ifdef _WIN32 // Windows 
class Timer
{
public:
	Timer()
		:	freq_(0.0), t_start_(0), start_was_called_(false)
	{}
	
	/**
	 *	Start the timer
	 */
	inline void start()
	{
		LARGE_INTEGER lpFrequency;

		// Get CPU frequency [counts/sec]
		if(!QueryPerformanceFrequency(&lpFrequency))
			FATAL_ERROR("QueryPerformanceFrequency failed");	
		freq_ = double(lpFrequency.QuadPart);

		// Get current performance count [counts]
		if(!QueryPerformanceCounter(&lpFrequency))
			FATAL_ERROR("QueryPerformanceCounter failed");

		start_was_called_ = true;
		t_start_ = lpFrequency.QuadPart; 
	}
	
	/**
	 *	Stop the timer and update total time
	 *	@return	the time in seconds since start()
	 */
	inline double stop()
	{
		if(!start_was_called_) 
			WARNING("calling stop() without previously calling start()");

		LARGE_INTEGER lpFrequency;
		QueryPerformanceCounter(&lpFrequency);
		__int64 t_end = lpFrequency.QuadPart;

		start_was_called_ = false;
		return double(t_end - t_start_)/freq_;
	}
	
private:
	double  freq_;
	__int64	t_start_;
	bool start_was_called_;
};

#else // Linux / Mac OSX 
class Timer
{
public:
	Timer()
		: t_start_(0.0), start_was_called_(false) 
	{
    	gettimeofday(&t_, NULL);
	}
	
	/**
	 *	Start the timer
	 */
	inline void start()
	{
		gettimeofday(&t_, NULL);
		t_start_ = t_.tv_sec + (t_.tv_usec/1000000.0);

		start_was_called_ = true;
	}
	
	/**
	 *	Stop the timer and update total time
	 *	@return	The time in seconds since start()
	 */
	inline double stop()
	{
		if(!start_was_called_) 
			WARNING("calling stop() without previously calling start()");

		gettimeofday(&t_, NULL);
		double t_cur =  t_.tv_sec + (t_.tv_usec/1000000.0) - t_start_;

		start_was_called_ = false;
		return t_cur;
	}

private:
	struct timeval t_;	
	double t_start_;
	bool start_was_called_;
};

#endif /* Linux / Mac OSX */

#endif /* C++11 */

class Progressbar
{
public:

	/**
	 *	Constructor
	 *	@param   max_it      maximum iterations until the progressbar is at 100%
	 */
	Progressbar(std::size_t max_it)
		: max_it_(max_it), progress_(0), cur_iter_(0), step_size_(1),
		  bar_width_(40)
	{}


	/**
	 *	Advance the Progressbar by one step. This will redraw the progressbar.
	 */
	inline void progress()
	{
		cur_iter_ += step_size_;
		std::size_t cur_pos = std::size_t( 100 * double(cur_iter_) / max_it_);
		
		if(cur_pos > progress_)
		{
			progress_ += cur_pos - progress_;

			// Print progressbar
			std::cout << " [";
			double pos = double(bar_width_ * progress_)/100;
			for (std::size_t i = 0; i < bar_width_; ++i) 
			   std::cout << (i < pos ? "#" : " ");
			std::cout << "] ";
		
			// Print Percentage
			std::cout.width(5);
			std::cout << progress_ << std::right;
			std::cout << " %\r";
			std::cout.flush();
		}
	}
	
	/**
	 *	Advance progressbar by multiple steps
	 *	@param n_steps  number of advanced steps
	 */
	inline void progress(int n_steps)
	{
		for(int i = 0; i < n_steps; ++i)
			progress();
	}
			
	/**
	 *	Pause the progressbar to allow other routines to print to std::cout
	 */
	inline void pause()
	{
		for(std::size_t i = 0; i < bar_width_ + 11; ++i) 
			std::cout << " ";
		std::cout << "\r";
		std::cout.flush(); 
	}

	// === Setter ===
	void set_step_size(std::size_t step_size) { step_size_ = step_size; }
	void set_bar_width(std::size_t bar_width) { bar_width_ = bar_width; }

private:
	std::size_t max_it_;
	
	std::size_t progress_;
	std::size_t cur_iter_;
	std::size_t step_size_;

	std::size_t bar_width_;
};

#endif /* utility.hpp */
