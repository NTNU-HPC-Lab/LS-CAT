/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  QLBopt is used to set various options of the QLB class.
 *  The options are:
 *   - plot        Unsigned integer where the bits indicate which quantities are
 *                 written to file after calling 'QLB::write_content_to_file()'
 *                 all           <==>        1
 *                 spread        <==>        2
 *                 spinor1       <==>        4
 *                 spinor2       <==>        8
 *                 spinor3       <==>       16
 *                 spinor4       <==>       32
 *                 density       <==>       64
 *                 currentX      <==>      128
 *                 currentY      <==>      256
 *                 veloX         <==>      512
 *                 veloY         <==>     1024
 *                 e.g to write spinor1 and currentX to file pass:
 *			       unsigned int plot = QLBopt::spinor1 | QLBopt::currentX;
 *  - verbose      Enables verbose mode to get some additional information written
 *                 to STDOUT during the simulation.
 *  - device       Set the device the simulation will run on
 *                 0 :   CPU serial
 *                 1 :   CPU multi threaded
 *                 2 :   GPU (CUDA)
 *  - tnhreads     Number of threads used by the CPU implementation.
 *	- config_file  Name of the configuration file.
 *  - g            Coupling constant of the built-in potentials.
 */

#ifndef QLB_OPT_HPP
#define QLB_OPT_HPP

#include <string>

class QLBopt
{
public:

	enum plot_t {     all = 1 << 0,    spread = 1 << 1,   spinor1 = 1 << 2,
	              spinor2 = 1 << 3,   spinor3 = 1 << 4,   spinor4 = 1 << 5,
	              density = 1 << 6,  currentX = 1 << 7,  currentY = 1 << 8,
	                veloX = 1 << 9,     veloY = 1 << 10  };
	
	// === Constructor ===
	QLBopt()
		:	plot_(0), verbose_(false), device_(0), nthreads_(1), config_file_(""),
		    g_(1.0)
	{}
	
	QLBopt(unsigned int plot, bool verb, int device, unsigned int nthreads,
	       std::string config_file, float g)
		:	plot_(plot), verbose_(verb), device_(device), nthreads_(nthreads),
			config_file_(config_file), g_(g)
	{}
	
	QLBopt(const QLBopt& opt)
		:	plot_(opt.plot()), verbose_(opt.verbose()), device_(opt.device()),
			nthreads_(opt.nthreads()), config_file_(opt.config_file()), g_(opt.g())
	{}
	
	// === Getter ===
	inline unsigned int plot() const { return plot_; }
	inline bool verbose() const { return verbose_; }
	inline int device() const { return device_; }
	inline unsigned int nthreads() const { return nthreads_; }
	inline std::string config_file() const { return config_file_; }
	inline float g() const { return g_; }

	// === Setter ===
	inline void set_plot(unsigned int plot) { plot_ = plot; }
	inline void set_verbose(bool verbose) { verbose_ = verbose; }
	inline void set_device(int device) { device_ = device; }
	inline void set_nthreads(unsigned int nthreads) { nthreads_ = nthreads; }
	inline void set_config_file(std::string file) { config_file_ = file; }
	inline void set_g(float g) { g_ = g; }

private:
	unsigned int plot_;
	bool verbose_;
	int device_;
	unsigned int nthreads_;
	std::string config_file_;
	float g_;
};

#endif /* QLBopt.hpp */
