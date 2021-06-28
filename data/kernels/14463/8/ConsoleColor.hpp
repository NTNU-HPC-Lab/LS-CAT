/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Coloring of terminal messages (Windows/UNIX)
 *  Usedy by 'error.hpp' and 'cuassert.hpp'
 */
 
#ifndef CONSOLE_COLOR_HPP
#define CONSOLE_COLOR_HPP

#include <iostream>
#include <vector>
#include <exception>
#include <cstdlib>

#ifdef _WIN32
 #include <windows.h>
#else 
 #include <string>
 #include <unistd.h>
#endif 

#ifdef _WIN32

class ConsoleColor
{
public:
	enum color_t { COLOR_RED=0, COLOR_GREEN=1, COLOR_MAGENTA=2, COLOR_WHITE=3 };

	/**
	 *	Initialize the ConsoleColor's
	 */
	ConsoleColor()
	{
		hstdout_ = GetStdHandle(STD_OUTPUT_HANDLE);
		GetConsoleScreenBufferInfo(hstdout_, &console_state_ );	
	
		color_table_.push_back(0x0C); // red
		color_table_.push_back(0x0A); // green
		color_table_.push_back(0x0D); // magenta
		color_table_.push_back(0x0F); // strong white
	}
	
	~ConsoleColor() 
	{
		reset_color();
	}
	
	/**
	 *	Set the console color
	 *	@param colour	0: red
	 *	                1: green
	 *	                2: magenta
	 *	                3: strong white
	 */
	void set_color(int color)
	{
		// We don't do anything if color index is out of bounds
		try
		{
			SetConsoleTextAttribute(hstdout_, color_table_.at(color));
		}
		catch(...) {}
	}
	
	/**
	 *	Reset the console color to the state before calling the constructor or
	 *	to black if no changes occured
	 */
	void reset_color()
	{	
		SetConsoleTextAttribute(hstdout_, console_state_.wAttributes );
	}
	
private:
	HANDLE hstdout_;	
	CONSOLE_SCREEN_BUFFER_INFO console_state_;

	std::vector<WORD> color_table_;
};

#else /* UNIX */ 

class ConsoleColor
{
public:
	enum color_t { COLOR_RED=0, COLOR_GREEN=1, COLOR_MAGENTA=2, COLOR_WHITE=3 };

	/**
	 *	Initialize the ConsoleColor's
	 */
	ConsoleColor()
	{
		is_terminal_ = isatty(STDOUT_FILENO);
		
		color_table_.push_back("\x1b[1;31m"); // red
		color_table_.push_back("\x1b[1;32m"); // green
		color_table_.push_back("\x1b[1;35m"); // magenta
		color_table_.push_back("\x1b[1m");    // strong white
	}
	
	~ConsoleColor() 
	{
		reset_color();
	}
	
	void set_color(int color)
	{
		if(is_terminal_)
		{
			try
			{
				std::cout << color_table_.at(color) << std::flush;
				std::cerr << color_table_.at(color) << std::flush;		
			}
			catch(...) {}
		}
	}
	
	/**
	 *	Reset the console color to black
	 */
	void reset_color()
	{	
		if(is_terminal_)
		{
			std::cout << "\x1b[0m" << std::flush;
			std::cerr << "\x1b[0m" << std::flush;
		}
	}
	
private:
	bool is_terminal_;
	
	std::vector<std::string> color_table_;
};

#endif

#endif /* ConsoleColor.hpp */
