/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  Parse potential and initial condition passed by the options '--potential=FILE' 
 *  and '--initial=FILE'.
 *	For the exact formatting of the files take a look at 'InputGenerator.py'
 */

#ifndef QLB_PARSER_HPP
#define QLB_PARSER_HPP

#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <complex>

#include "CmdArgParser.hpp"
#include "error.hpp"

class QLBparser
{
public:
	/**
	 *	Open the filestreams for the potential and input file
	 *	@param 	potential_file   file with potential data (or empty)
	 *	@param 	initial_file     file with inital data (or empty)
	 */
	QLBparser(std::string potential_file, std::string initial_file);
	
	/**
	 *	Copy constructor
	 */
	QLBparser(const QLBparser& parser);
	
	/**
	 *	Close all open filestreams
	 */
	~QLBparser();

	/**
	 *	Parse the content of the input files (if present)
	 *	@param cmd   parsed command line arguments
	 */
	void parse_input(const CmdArgParser* cmd);

	// === Getter ===
	inline unsigned L() const { return L_; }
	inline float dx() const { return dx_;}
	inline bool mass_is_present() const { return mass_is_present_; }
	inline float mass() const { return mass_; }
	inline bool delta0_is_present() const { return delta0_is_present_; }
	inline float delta0() const { return delta0_; }

	inline bool potential_is_present() const { return potential_is_present_; }
	inline bool initial_is_present() const { return initial_is_present_; }
	inline std::string potential_file() const { return potential_file_; }
	inline std::string initial_file() const { return initial_file_; }
	
	inline bool is_valid() const { 
		return potential_is_present_ || initial_is_present_; }
	
	std::vector< float > potential_;
	std::vector< std::complex<float> > initial_;
	
private:
	bool potential_is_present_;
	bool initial_is_present_;
	
	unsigned L_;
	float dx_;
	
	bool mass_is_present_;
	float mass_;
	
	bool delta0_is_present_;
	float delta0_;
	
	std::string initial_file_;
	std::string potential_file_;
	
	std::ifstream pfin;
	std::ifstream ifin;
};

#endif /* QLBparser.hpp */
