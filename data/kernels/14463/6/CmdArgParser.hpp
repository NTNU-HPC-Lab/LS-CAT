/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  This file contains the class "CmdArgParser" and it's friends which are 
 *  used to parse command-line input passed to the function main(...).
 *
 *  [EXAMPLE]
 *  The execution of the main program could look as follows:
 * 
 *  ./main --foo
 * 
 *  The class CmdArgParser will then look for "--foo" and set the internal
 *  variables affiliated with this argument accordingly. The class supports 
 *  three types of arguments "--foo", "--foo=X" where X is some numerical 
 *  constant and "--foo=S" where S is a string. 
 */

#ifndef CMD_ARG_PARSER_HPP
#define CMD_ARG_PARSER_HPP

// System includes 
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <thread>
#include <algorithm>
#include <exception>

// Compiler hints
#ifdef _WIN32
 #undef min
 #undef max
 #define NO_RETURN	__declspec(noreturn)
#else
 #define NO_RETURN	__attribute__((noreturn))
#endif

// Detect compiler
#if defined(__clang__)
 #define COMPILER 	"Clang/LLVM "
 #define VERSION 	__clang_version__
#elif defined(__ICC) || defined(__INTEL_COMPILER)
 #define COMPILER	"ICC/ICPC "
 #define VERSION	__VERSION__
#elif defined(__GNUC__) || defined(__GNUG__)
 #define COMPILER	"GNU GCC/G++ "
 #define VERSION	__VERSION__
#elif defined(_MSC_VER) && defined(_WIN32)
 #define COMPILER	"Microsoft(R) Visual C/C++ Compiler "
 #define VERSION	_MSC_VER
#else
 #define COMPILER	"unknown compiler "
 #define VERSION	""
#endif

// Detect architecture
#if defined(__x86_64__) || defined(_M_X64)
 #define ARCH	"x86 64-bit (amd64)"
#elif defined(__i386) || defined(_M_IX86)
 #define ARCH	"x86 32-bit (i386)"
#else
 #define ARCH	"unknown architecture"
#endif

class CmdArg
{
public:
	/** 
	 *	Represent a single command-line argument which is of the form "--command"
	 *	@param is_present    boolean whether the command-line argument was passed.
	 */
	CmdArg(bool is_present) : is_present_(is_present) {}

	inline bool is_present() const { return is_present_; }
	inline void set_is_present(bool is_present) { is_present_ = is_present; }

private:
	bool is_present_;
};


template< typename value_t >
class CmdArgNumeric : public CmdArg
{
public:
	/** 
	 *	Represent a single command-line argument which is of the form "--command=X"
	 *	@param is_present    boolean whether the command-line argument 
	 *	                     was passed.						
	 *	@param value         value of the numerical constant (if present).
	 */
	CmdArgNumeric(bool is_present, value_t value)
		: CmdArg(is_present), value_(value)
	{}

	inline value_t value() const { return value_; }
	inline void set_value(value_t value) { value_ = value; }
	
private:
	value_t value_;
};


class CmdArgString : public CmdArg
{
public:
	/** 
	 *	Represent a single command-line argument which is of the form "--command=S"
	 *	@param is_present	boolean whether the command-line argument 
	 *						was passed.						
	 *	@param str			value of the string (if present) 
	 */
	CmdArgString(bool is_present, std::string str)
		: CmdArg(is_present), str_(str)
	{}

	inline std::string str() const { return str_; }
	inline void set_str(std::string str) { str_ = str; }
	
private:
	std::string str_;
};


class CmdArgException : public std::exception
{
public:
	/**
	 *	Constructor - Handle exception
	 *	@param	msg		warning message
	 */
	CmdArgException(std::string msg)
		: msg_(msg)
	{}
	
	virtual ~CmdArgException() throw() {}

	/**
	 *	Exception handling
	 *	@return warning message passed during throw
	 */
	virtual const char* what() const throw()
	{
		return msg_.c_str(); 
	}
	
private:
	std::string msg_;
};


class CmdArgParser
{
public:
	typedef std::vector<std::string>::iterator iterator_t;
	typedef std::vector<std::string> svec_t;

	/**
	 *	Parse all arguments in argv
	 *	@param	argc	length of argv
	 *	@param	argv	argument array passed to main(int argc, char* argv[])
	 */
	CmdArgParser(int argc, char* argv[])
		: argv_(argv+1,argv+argc), program_(argv[0]), 
		  width_cmd(20), width_expl(45)
	{
		// === Parse arguments ===
		try
		{ 					
			// --help
			if(find("--help").is_present()) print_usage();

			// --version
			if(find("--version").is_present()) print_version();
		
			// --verbose
			verbose_ = find("--verbose").is_present();
			
			// --L=X
			auto L = find_numeric<int>("--L");
			L_ = L.is_present();
			if(L_ && L.value() < 2) 
				throw CmdArgException("'X' must be > 1 in command-line argument"
				                      " '--L=X'");
			L_value_ = L.value();
			
			// --dx=X
			auto dx = find_numeric<float>("--dx");
			dx_ = dx.is_present();
			if(dx_ && dx.value() < 0) 
				throw CmdArgException("'X' must be > 0 in command-line argument"
				                      " '--dx=X'");
			dx_value_ = dx.value();
			
			// --dt=X
			auto dt = find_numeric<float>("--dt");
			dt_ = dt.is_present();
			if(dt_ && dt.value() < 0) 
				throw CmdArgException("'X' must be > 0 in command-line argument"
				                      " '--dt=X'");
			dt_value_ = dt.value();
			
			// --delta0=X
			auto delta0 = find_numeric<float>("--delta0");
			delta0_ = delta0.is_present();
			if(delta0_ && delta0.value() < 0) 
				throw CmdArgException("'X' must be > 0 in command-line argument"
				                      " '--delta0=X'");
			delta0_value_ = delta0.value();
			
			// --mass=X
			auto mass = find_numeric<float>("--mass");
			mass_ = mass.is_present();
			if(mass_ && mass.value() < 0) 
				throw CmdArgException("'X' must be > 0 in command-line argument"
				                      " '--mass=X'");
			mass_value_ = mass.value();
			
			// --tmax=X
			auto tmax = find_numeric<int>("--tmax");
			tmax_ = tmax.is_present();
			if(tmax_ && tmax.value() < 0) 
				throw CmdArgException("'X' must be > 0 in command-line argument"
				                      " '--tmax=X'");
			tmax_value_ = tmax.value();
			
			// --no-gui
			gui_ = !find("--no-gui").is_present();
				
			// --potential=FILE or [harmonic|free|barrier]
			auto potential = find_string("--potential");
			potential_ = 0;
			potential_file_ = "";
			if(potential.is_present() && compare(potential.str(),"free"))
				potential_ = 0;
			else if(potential.is_present() && compare(potential.str(),"harmonic"))
				potential_ = 1;
			else if(potential.is_present() && compare(potential.str(),"barrier"))
				potential_ = 2;
			else if(potential.is_present() && compare(potential.str(),"GP"))
				potential_ = 3;
			else if(potential.is_present())
			{
				potential_ = 4;
				potential_file_ = potential.str();
			}

			// --g=X
			auto g = find_numeric<float>("--g");
			g_ = g.is_present();
			g_value_ = g.value();

			// --initial=S
			auto initial = find_string("--initial");
			initial_file_ = initial.is_present() ? initial.str() : "";
			
			// --config=S
			auto config = find_string("--config");
			config_file_ = config.is_present() ? config.str() : "QLBconfig.conf";
				                     
			// --device=[cpu-serial|cpu-thread|gpu]
			auto device = find_string("--device");
#ifdef QLB_HAS_CUDA
			device_ = 2;
#else
			device_ = 1;
#endif
			if(device.is_present() && compare(device.str(),"cpu-serial"))
				device_ = 0;
			else if(device.is_present() && compare(device.str(),"cpu-thread"))
				device_ = 1;
			else if(device.is_present() && compare(device.str(),"gpu"))
				device_ = 2;	
			else if(device.is_present())
				throw CmdArgException("'S' in '--device=S' must be one of "
				                      "[cpu-serial|cpu-thread|gpu]");
			
			// --nthreads=X
			auto nthreads = find_numeric<int>("--nthreads");
			nthreads_ = nthreads.is_present();
			if(nthreads_ && nthreads.value() < 1)
				throw CmdArgException("'X' must be > 0 in command-line argument"
				                      " '--nthreads=X'");
			
			max_threads_ = std::thread::hardware_concurrency();
		 	if(device_ == 1 && !nthreads_)
		 	{
		 		nthreads_ = true;	
		 		if(gui_ == 1)
		 		{
					nthreads_value_ =  L_value_ > 256 ? max_threads_ - 1 : 
					                                    max_threads_/2;
				}
				else
					nthreads_value_ =  max_threads_ / 2;
			}
			else if(nthreads_ && nthreads.value() == 1)
			{
				device_ = 0;
				nthreads_value_ = nthreads.value();
			}
			else if(nthreads_)
			{
				device_ = 1;
				nthreads_value_ = nthreads.value();
			}
			else
				nthreads_value_ = 1;
			
			// --plot=[all,spread,spinor1,spinor2,spinor3,spinor4,density,
			//         currentX,currentY,veloX,veloY]
			auto plot = find_string("--plot");
			plot_ = 0;
			if(plot.is_present())
				set_plot_args(plot.str());
											
			// --fullscreen
			fullscreen_ = find("--fullscreen").is_present();
			if(fullscreen_ && !gui_)
				gui_ = true;
				
			// --dump-at=X
			auto dump = find_numeric<int>("--dump-at");
			dump_ = dump.is_present();
			if(dump_ && dump.value() < 0) 
				throw CmdArgException("'X' must be non-negativ in command-line "
				                      "argument '--dump-at=X'");
			dump_value_ = dump.value();
			
			// --dump-load=S
			auto static_viewer = find_string("--dump-load");
			static_viewer_ = static_viewer.is_present();
			static_viewer_file_ = static_viewer.str();
			if(static_viewer_)
				gui_ = true;
		
			// --start-rotating
			start_rotating_ = find("--start-rotating").is_present();
	
			// --start-paused
			start_paused_ = find("--start-paused").is_present();
			
			// --disable-progressbar
			progressbar_ = !find("--disable-progressbar").is_present();
			
			auto scaling = find_numeric<int>("--start-scaling");
			start_scaling_ = scaling.is_present() ? scaling.value() : 1;
			
			// --disable-light
			light_ = !find("--disable-light").is_present();
		
			// Throw if we have unparsed arguments
			if(argv_.size())
			{
				std::string err_msg = "unrecognized option ";
				for(std::size_t i = 0; i < argv_.size(); ++i)
				{
					err_msg += "'"+argv_[i]+"'";
					throw CmdArgException(err_msg + suggest_action(argv_[i]));		
				}
			}
		}
		catch(std::exception& e)
		{	
			std::cerr <<"QLB: error: " <<  e.what() << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	// === Access Arguments ===
	inline bool verbose() const { return verbose_; }
	inline bool fullscreen() const { return fullscreen_; }
	inline bool nthreads() const { return nthreads_; }
	inline unsigned nthreads_value() const { return nthreads_value_; }
	inline unsigned max_threads() const { return max_threads_; }
	inline bool L() const { return L_; }
	inline unsigned L_value() const { return L_value_; }
	inline bool dx() const { return dx_; }
	inline float dx_value() const { return dx_value_; }
	inline bool dt() const { return dt_; }
	inline float dt_value() const { return dt_value_; }
	inline bool delta0() const { return delta0_; }
	inline float delta0_value() const { return delta0_value_; }
	inline bool mass() const { return mass_; }
	inline float mass_value() const { return mass_value_; }
	inline bool tmax() const { return tmax_; }
	inline unsigned tmax_value() const { return tmax_value_; }
	inline bool dump() const { return dump_; }
	inline unsigned dump_value() const { return dump_value_; }
	inline bool g() const { return g_; }
	inline float g_value() const { return g_value_; }
	inline bool static_viewer() const { return static_viewer_; }
	inline std::string static_viewer_file() const { return static_viewer_file_; }
	inline int potential() const { return potential_; }
	inline std::string potential_file() const { return potential_file_; }
	inline std::string initial_file() const { return initial_file_; }
	inline std::string config_file() const { return config_file_; }
	inline bool gui() const { return gui_; } 
	inline unsigned int plot() const { return plot_; }
	inline bool progressbar() const { return progressbar_; }
	inline int device() const { return device_; }
	
	// undocumented
	inline bool start_rotating() const { return start_rotating_; }
	inline bool start_paused() const { return start_paused_; }
	inline int  start_sclaing() const { return start_scaling_; }
	inline bool light() { return light_; }
	
private:

	/**
	 *	Search the command line argument list for "--command"
	 *	@param  command		searched for string
	 *	@return parse the command into a CmdArg object 
	 */
	inline CmdArg find(std::string command)
	{
		args_looked_for_.push_back(command);
		CmdArg foundArg(false);
		
		// Loop over the array and look for the string
		for(iterator_t it = argv_.begin(); it != argv_.end(); )
		{
			if(it->compare(command) == 0) 
			{
				it = argv_.erase(it);
				foundArg.set_is_present(true);
			}
			else 
				++it;
		}
		return foundArg;		 
	}

	/**
	 *	Search the command line argument list for "--command=X" where  X is some
	 *	numerical constant. The template argument of the function is the type of
	 *	the constant X.
	 *	@param  command   searched for string
	 *	@return parse the command into a CmdArgNumeric object 
	 */
	template< typename numeric_t > 
	inline CmdArgNumeric<numeric_t> find_numeric(std::string command)
	{
		args_looked_for_.push_back(command+"=X");
		
		CmdArgNumeric<numeric_t> foundArg(false,0);
		
		// Loop over the array and look for "command"
		for(iterator_t it = argv_.begin(); it != argv_.end(); )
		{
			if(compare(command, *it)) 
			{
				std::string cmd = *it;
				it = argv_.erase(it);
			
				std::size_t delimiter_pos = cmd.find("=");
					
				// Delimiter "=" in "--command=X" is missing
				if(delimiter_pos == std::string::npos)
				{
					std::string err_msg = "missing delimiter '=' in ";
					err_msg += "command-line argument '"+command+"=X'";
					throw CmdArgException(err_msg);
				}

				// X in "--command=X" is missing
				if(delimiter_pos == cmd.size()-1)
				{
					std::string err_msg = "missing 'X' in ";
					err_msg += "command-line argument '"+command+"=X'";
					throw CmdArgException(err_msg);
				}
					
				// Extract X from cmd as a string
				std::string X(cmd,delimiter_pos+1,cmd.size());
					
				// Read X into value
				std::istringstream value_str(X);
				numeric_t value;
				value_str >> value;
										
				foundArg.set_is_present(true);
				foundArg.set_value(value);					
			}
			else
				++it;
		}
		return foundArg;
	}

	/**
	 *	Search the command line argument list for "--command=S" where S is a
	 *	string
	 *	@param  command   searched for string
	 *	@return parse the command into a CmdArgString object 
	 */ 
	inline CmdArgString find_string(std::string command)
	{
		args_looked_for_.push_back(command+"=S");
		CmdArgString foundArg(false,"");
		
		// Loop over the array and look for "command"
		for(iterator_t it = argv_.begin(); it != argv_.end(); )
		{
			if(compare(command, *it)) 
			{
				std::string cmd = *it;
				it = argv_.erase(it);
			
				std::size_t delimiter_pos = cmd.find("=");
					
				// Delimiter "=" in "--command=S" is missing
				if(delimiter_pos == std::string::npos)
				{
					std::string err_msg = "missing delimiter '=' in ";
					err_msg += "command-line argument '"+command+"=S'";
					throw CmdArgException(err_msg);
				}
		
				// S in "--command=S" is missing
				if(delimiter_pos == cmd.size()-1)
				{
					std::string err_msg = "missing 'S' in ";
					err_msg += "command-line argument '"+command+"=S'";
					throw CmdArgException(err_msg);
				}
					
				// Extract S from cmd as a string
				std::string S(cmd,delimiter_pos+1,cmd.size());
										
				foundArg.set_is_present(true);
				foundArg.set_str(S);					
			}
			else
				++it;
		}
		return foundArg;
	}

	
	/**
	 *	String comparison between cmd1 and cmd2. The method will yield true if 
	 *	"--cmd1" == "--cmd2" or "--cmd1" == "--cmd2=X" 
	 *	@param	cmd1	Command searched for
	 *	@param 	cmd2	Command in the argv_ array
	 *	@return true if the strings match, false otherwise
	 */
	inline bool compare(std::string cmd1, std::string cmd2) const
	{
		std::size_t delimiter_pos = cmd2.find("=");

		if(delimiter_pos == std::string::npos)
			return cmd1.compare(cmd2) == 0;
		else
		{
			std::string cmd2_extracted(cmd2,0,delimiter_pos);
			return cmd1.compare(cmd2_extracted) == 0;
		}
	}		
	
	/**
	 *	Print a help line consisting of the command and corresponding
	 *	explanation
	 *	@param	cmd		Command-line argument to trigger the command
	 *	@param 	expl	Explanation of the command (single line)
	 */
	inline void print_help_line(std::string cmd, std::string expl) const
	{
		std::cout << std::setw(width_cmd) << std::left << "    "+cmd
		          << std::setw(width_expl) << expl << std::endl;
	}

	/**
	 *	Print multiple help lines consisting of the command and corresponding
	 *	explanation seprated by multiple lines.
	 *	@param	cmd		Command-line argument to trigger the command
	 *	@param 	expl	Explanation of the command (each element is a new line)
	 */
	inline void print_help_line(std::string cmd, svec_t expl) const
	{
		std::cout << std::setw(width_cmd) << std::left << "    "+cmd
		          << std::setw(width_expl) << expl[0] << std::endl;
		for(iterator_t it = expl.begin()+1; it != expl.end(); ++it)
			std::cout << std::setw(width_cmd) << std::left << " "
					  << std::setw(width_expl) << *it << std::endl;
	}
	
	/**
	 *	Print the usage of this program (triggered by --help)
	 */
	NO_RETURN void print_usage() const
	{
		std::cout << "  Usage : " << std::endl;
		std::cout << "     " << program_ << " [options ... ] " << std::endl;
		std::cout << std::endl;
		std::cout << "  Options :" << std::endl;
		
		// Print all available options
		print_help_line("--help","Display this information");
		print_help_line("--version","Display version information");
		print_help_line("--verbose","Print statistics about the OpenGL and CUDA runtime");
		print_help_line("--no-gui","Turn off visualization");
		print_help_line("--fullscreen","Start in fullscreen mode (if possible)");
		std::string expl_potential[2] = {
		             "Set the potential to S, where S is either an input file",
                     "(see InputGenerator.py) or one of [free|harmonic|barrier|GP]"};
		print_help_line("--potential=S", svec_t(expl_potential, expl_potential+2));
		print_help_line("--g=S","Set the coupling constant of the potentials [default 1.0]");
		std::string expl_intial[4] = {
		            "Set the initial condition to S, where S an input file",
                    "(see InputGenerator.py or InputGenerator.m). By default a", 
					"gaussian with spread delta0 is being used which can be ",
					"controlled by '--delta0=X'"};
		print_help_line("--initial=S", svec_t(expl_intial, expl_intial+4));
		print_help_line("--tmax=X","Run the simulation in the interval [0, X*dt]" );
		std::string expl_X[2] = {
			             "Set the number of grid points in each dimension",
	                     "to X [default: 128]"};
		print_help_line("--L=X", svec_t(expl_X, expl_X+2));
		print_help_line("--dx=X","Set spatial discretization to X [default: 1.5625]");
		print_help_line("--dt=X","Set temporal discretization to X [default: 1.5625]");
		print_help_line("--delta0=X","Set initial spread to X [default: 14.0]");
		print_help_line("--mass=X","Set mass of the particles to X [default: 0.1]");
		std::string expl_plot[4] = {"Specify which quantities are written to a file,"
		                           " where S","can be a combination of the following"
		                           " (delimited with ',')",
		                           "[all, spread, spinor1, spinor2, spinor3, spinor4,"
		                           " density,"," currentX, currentY, veloX, veloY]"};
		print_help_line("--plot=S",svec_t(expl_plot, expl_plot+4));
		std::string expl_device[2] = {"Set the device the simulation will run on, S must be",
		                              "one of [cpu-serial|cpu-thread|gpu]"};
		print_help_line("--device=S",svec_t(expl_device, expl_device+2));
		print_help_line("--nthreads=X","Execute CPU verison with X threads");
		print_help_line("--config=S", "Set the configuration file to S [default: QLBconfig.conf]");
		print_help_line("--dump-at=X","Dump the state of the simulation at time X*dt to "
		                "a file" );
		print_help_line("--dump-load=S","Load the dump file S to be used with the static viewer");
		exit(EXIT_SUCCESS);
	}

	/**
	 *	Print the version information of this program (triggered by --version)
	 */
	NO_RETURN void print_version() const
	{
		std::cout << "QLB - 1.0" << std::endl;
		std::cout << "Built on " << __TIMESTAMP__ << " for " << ARCH << std::endl;
		std::cout << "Compiled with " << COMPILER << VERSION << std::endl;
		std::cout << "Compute model : CPU";
#ifdef QLB_HAS_CUDA
		std::cout << ", CUDA";
#endif
		std::cout << std::endl;
		
		exit(EXIT_SUCCESS);
	}


	/**
	 *	Set the bits to indicate which files need to be plotted by searching 
	 *	'str' for known arguments: 
	 *	[all,spread,spinor1,spinor2,spinor3,spinor4,density,currentX,currentY,
	 *	 veloX,veloY]
	 *	@param 		str		string of commands (non empty)
	 */
	void set_plot_args(std::string str)
	{

		const std::string known_args[11] = {"all","spread","spinor1","spinor2",
		                                    "spinor3","spinor4","density",
		                                    "currentX","currentY","veloX","veloY"};
		const std::size_t nknown_args = 11;
		
		bool match = false;
		std::size_t delimiter_pos = str.find(",");
		std::string command;
		std::string str_backup = str;
		
		if(delimiter_pos == (str.size()-1))
			throw CmdArgException("command-line argument '--plot="+str_backup+
			                      "' ends with delimiter ','");
		do
		{
			match = false;
			
			if(delimiter_pos == std::string::npos)
			{ 
				command = str;
				delimiter_pos = str.size()-1;
			}
			else
				command = str.substr(0, delimiter_pos);

			for(std::size_t i = 0; i < nknown_args; ++i)
				if(command.compare(known_args[i]) == 0)
				{
					match = true;
					plot_ |= 1 << i;
				}

			if(!match)
				throw CmdArgException("'"+command+"'"+" in command-line "
				                      "argument '--plot="+str_backup+
				                      "' is not known");
			// Remove command from str
			str = str.substr(delimiter_pos+1, str.size());
			delimiter_pos = str.find(",");
			
		} while(str.size() != 0);
	}	

	/**
	 *	If we have unparsed arguments i.e the commands are not known
	 *	we make a suggestion for the command or advise the user to use "--help".
	 *	To find the closest matching string the Levenshtein distance between
	 *	the unknown command and the searched for commands is calculated.
	 *
	 *	@param	str		unknown command
	 */
	inline std::string suggest_action(std::string str) const
	{
		std::size_t delimiter_pos = str.find("=");
		std::string str_after_delimiter = "";
		std::string str_unknown = str;
		
		// If there is a delimiter we save the value after and add "=X" to the
		// command i.e we look for "command=X"
		if(delimiter_pos != std::string::npos)
		{
			str_after_delimiter = std::string(str, delimiter_pos, str.size());
			str_unknown = std::string(str, 0, delimiter_pos)+"=X";
		}

		unsigned indx = 0;
		const unsigned dist_limit = 3;
		unsigned cur_dist = dist_limit, dist = 0;
		
		// Compare with all the searched for commands
		for(unsigned i = 0; i < args_looked_for_.size(); ++i)
		{
			dist = levenshtein_dist(str_unknown, args_looked_for_[i]); 
			if(dist < std::min(dist_limit,cur_dist))
			{
				cur_dist = dist;
				indx = i;
			}
		}
		
		// Make a suggestion 
		std::string suggestion;
		if(cur_dist < dist_limit)
		{
			// Remove the "=X" if we added it earlier
			std::string str_suggest = args_looked_for_[indx];
			if(str_after_delimiter.size() != 0)
				str_suggest = std::string(str_suggest, 0, str_suggest.size()-2);
			suggestion = " did you mean '"+str_suggest+str_after_delimiter+"' ?";
		}
		else
			suggestion = " try '--help' for help";
		return suggestion;
	}

	/**
	 *	Compute the levenshtein distance between two strings. 
	 *	@Refrence http://en.wikipedia.org/wiki/Levenshtein_distance
	 *
	 *	@param	str1	first string
	 *	@param 	str2	second string
	 */
	unsigned levenshtein_dist(const std::string& s1,const std::string& s2) const
	{
		const std::size_t len1 = s1.size(); 
		const std::size_t len2 = s2.size();
		
		std::vector<unsigned> v0(len2+1), v1(len2+1);
	 
		for(unsigned i = 0; i < v1.size(); ++i)
			v1[i] = i;
			
		for(unsigned i = 0; i < len1; ++i) 
		{
			v0[0] = i+1;
			for(unsigned j = 0; j < len2; ++j)
			{
				int cost = (s1[i] == s2[j]) ? 0 : 1;
				v0[j+1] = std::min(std::min(v1[j+1] + 1, v0[j] + 1), v1[j] + cost);
			}
			v0.swap(v1);
		}
		return v1[len2];
	}

private:
	svec_t argv_;
	svec_t args_looked_for_;
	
	std::string program_;
	
	// === Arguments ===
	bool verbose_;
	bool fullscreen_;
	bool nthreads_;
	unsigned nthreads_value_;
	unsigned max_threads_;
	bool L_; 	 
	unsigned L_value_;
	bool dx_;
	float dx_value_;
	bool dt_;
	float dt_value_;
	bool delta0_;
	float delta0_value_;
	bool mass_;
	float mass_value_;
	bool tmax_;
	unsigned tmax_value_;
	bool dump_;
	unsigned dump_value_;
	bool g_;
	float g_value_;
	bool static_viewer_;
	std::string static_viewer_file_;
	bool gui_;
	int potential_;
	std::string potential_file_;
	std::string initial_file_;
	std::string config_file_;
	unsigned int plot_;
	int device_;
	bool start_rotating_;
	bool start_paused_;
	bool progressbar_;
	bool light_;
	int start_scaling_;
	
	// === IO ===
	std::size_t width_cmd;
	std::size_t width_expl;
};

#undef NO_RETURN
#undef VERSION
#undef COMPILER
#undef ARCH

#endif /* CmdArgParser.hpp */
