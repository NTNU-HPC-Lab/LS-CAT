/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Error handling code for fatal_error's and warning's.
 *	
 *  The functions should be called with FATAL_ERROR("message") and 
 *  WARNING("message").
 */

#ifndef ERROR_HPP
#define ERROR_HPP

// System includes
#include <iostream>
#include <algorithm>

// Local includes
#ifndef NO_COLOR_TERMINAL
 #include "ConsoleColor.hpp"
#endif

#ifdef _WIN32
 #define NO_RETURN	__declspec(noreturn)
#else
 #define NO_RETURN	__attribute__((noreturn))
#endif

// API to use the functions
#define FATAL_ERROR(msg) { fatal_error((msg), __FILE__, __LINE__); }
#define WARNING(msg)     { warning((msg), __FILE__, __LINE__); }

#ifdef _WIN32

struct MatchPathSeparator
{
    bool operator()(char ch) const
    {
        return ch == '\\' || ch == '/';
    }
};

#else

struct MatchPathSeparator
{
    bool operator()(char ch) const
    {
        return ch == '/';
    }
};

#endif

static inline std::string get_base_name(std::string pathname)
{
    return std::string(std::find_if(pathname.rbegin(), pathname.rend(),
                       MatchPathSeparator()).base(), pathname.end());
}

#ifndef NO_COLOR_TERMINAL

template< typename msg_t >
NO_RETURN static inline void fatal_error(const msg_t errmsg , const char *file, int line)
{
	ConsoleColor cc;
	cc.set_color(ConsoleColor::COLOR_WHITE);
	std::cerr << get_base_name(file).c_str() << ":" << line;
	cc.set_color(ConsoleColor::COLOR_RED);
	std::cerr << " error: ";
	cc.reset_color();
	std::cerr << errmsg << std::endl;
	exit(EXIT_FAILURE);
}

template< typename msg_t >
static inline void warning(const msg_t warnmsg , const char *file, int line)
{
	ConsoleColor cc;
	cc.set_color(ConsoleColor::COLOR_WHITE);
	std::cerr << get_base_name(file).c_str() << ":" << line;
	cc.set_color(ConsoleColor::COLOR_MAGENTA);
	std::cerr << " warning: ";
	cc.reset_color();
	std::cerr << warnmsg << std::endl;
}

#else

template< typename msg_t >
NO_RETURN static inline void fatal_error(const msg_t errmsg , const char *file, int line)
{
	std::cerr << get_base_name(file).c_str() << ":" << line;
	std::cerr << " error: ";
	std::cerr << errmsg << std::endl;
	exit(EXIT_FAILURE);
}

template< typename msg_t >
static inline void warning(const msg_t warnmsg , const char *file, int line)
{
	std::cerr << get_base_name(file).c_str() << ":" << line;
	std::cerr << " warning: ";
	std::cerr << warnmsg << std::endl;
}

#endif /* NO_COLOR_TERMINAL */

#endif /* error.hpp */
