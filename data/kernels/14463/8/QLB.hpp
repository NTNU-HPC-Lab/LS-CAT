/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zurich
 *
 *  Implementation of a Dirac Solver using the Quantum Lattice Boltzmann scheme.
 *  This file contains the class definition of QLB and all other
 *  definitions used in QLB*.cpp files.
 *
 *  @References
 *  Isotropy of three-dimensional quantum lattice Boltzmann schemes, 
 *  P.J. Dellar, D. Lapitski, S. Palpacelli, S. Succi, 2011
 */

#ifndef QLB_HPP
#define QLB_HPP

#define _USE_MATH_DEFINES

// System includes
#include <complex>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>		
#include <atomic>

// Local includes
#include "error.hpp"
#include "GLerror.hpp"
#include "utility.hpp"
#include "barrier.hpp"
#include "VBO.hpp"
#include "matrix.hpp"

#include "QLBopt.hpp"
#include "QLBparser.hpp"

#ifdef QLB_HAS_CUDA
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <device_launch_parameters.h>
 #include "cudaComplex.hpp"
 #include "cuassert.hpp"
#endif

#define QLB_MAJOR	1
#define QLB_MINOR	0

/********************************
 *  Quantum Lattice Boltzmann   *
 ********************************/
class QLB
{
public:

	// === typedefs ===
#ifdef QLB_SINGLE_PRECISION 
	typedef float                  float_t;
	typedef std::complex<float>    complex_t;
#else
	typedef double                 float_t;
	typedef std::complex<double>   complex_t;
#endif
 
	typedef matND<complex_t>       cmat_t;
	typedef matND<float_t>         fmat_t;
	typedef matN4D<complex_t>      c4mat_t;
	typedef SpinBarrier            barrier_t;
	
	enum scene_t  { spinor0 = 0, spinor1  = 1, spinor2  = 2, spinor3 = 3,
	                density = 4, currentX = 5, currentY = 6 };
	enum render_t { SOLID = GL_TRIANGLES, WIRE = GL_LINE_STRIP };

	// === Constants ===
	static const complex_t img;
	static const complex_t one;
	
	static const cmat_t X;
	static const cmat_t Y;

	static const cmat_t Xinv;
	static const cmat_t Yinv;
	
	static const cmat_t alphaX;
	static const cmat_t alphaY;

	static const cmat_t beta;
	
#ifdef QLB_HAS_CUDA 
	cuFloatComplex* d_X;
	cuFloatComplex* d_Y;
	
	cuFloatComplex* d_Xinv;
	cuFloatComplex* d_Yinv;
	
	cuFloatComplex* d_alphaX;
	cuFloatComplex* d_alphaY;
	
	cuFloatComplex* d_beta;
#endif 
	
	// === Constructor & Destructor ===
	
	/** 
	 *	Constructor 
	 *	@param	L       Lenght of the grid i.e grid will be L x L
	 *	@param 	dx      Spatial discretization
	 *	@param	mass    Mass of the particles
	 *	@param 	dt      Temporal discretization
	 *	@param  delta0  Initial spread
	 *	@param  tmax    Maximum time (only used if spreads are recoreded)
	 *	@param 	V_indx  Index of the potential function V (if no potential file
	 *	                is provided)
	 *	                0: no potential
	 *	                1: harmonic potential
	 *	                2: barrier potential
	 *	@param  parser  Class handling input files for potential and initial
	 *	                conditions (see at QLBparser 'QLBparser.hpp' )
	 *	@param  opt     Class defining options concerning plotting etc. 
	 *	                (see at QLBopt 'QLBopt.hpp' for further information)
	 *	@file 	QLB.cpp
	 */
	QLB(unsigned L, float_t dx, float_t mass, float_t dt, float_t delta0, 
	    unsigned tmax, int V_indx, QLBparser parser, QLBopt opt);
	
	/** 
	 *	Constructor (used by the StaticViewer)
	 *	@param  array_vertex  coordinates of the vertices
	 *	@param  array_normal  coordinates of the corresponding normals
	 *	@file 	QLB.cpp
	 */
	QLB(unsigned L, int V_indx, float_t dx, float_t mass, float_t scaling, 
        const std::vector<float>& array_vertex, 
        const std::vector<float>& array_normal, QLBopt opt);
	
	/** 
	 *	Destructor
	 *	@file 	QLB.cpp 
	 */
	~QLB();
	
	// === Initialization ===
	
	/** 
	 *	Allocates GPU memory needed during simulation
	 *	@file 	QLBcuda.cu	
	 */
	void allocate_device_arrays();
	
	/** 
	 *	Frees GPU memory used during simulation 	
	 *	@file 	QLBcuda.cu
	 */
	void free_device_arrays();
	
	/** 
	 *	init_device
	 *	@file 	QLBcuda.cu 
	 */
	void init_device();
	
	// === Initial condition ===
	
	/** 
	 *	Initial condition in which the positive energy, spin-up component is 
	 *	a spherically symmetric Gaussian wave packet with spread delta0.
	 *	spinor0 = C * exp( -( (x - x0)^2 + (y - y0)^2) / (4 * delta0^2) )
	 *	@param  i0    row index of the potential minimum in x-axis 
	 *	@param  j0    column index of the potential minimum in y-axis
	 *	@file 	QLB.cpp	
	 */
	void initial_condition_gaussian(int i0, int j0);

	// === Simulation ===
	
	/** 
	 *	Evolve the system for one time step (CPU - single threaded)
	 *	@file	QLBcpu.cpp	
	 */
	void evolution_CPU_serial();
	
	/** 
	 *	Evolve the system for one time step (CPU - multi threaded)
	 *	@param  tid     thread id in [0, nthreads)
	 *	@file	QLBcpu.cpp	
	 */
	void evolution_CPU_thread(int tid);

	/** 
	 *	Evolve the system for one time step (CUDA)
	 *	@file	QLBcuda.cpp	
	 */
	void evolution_GPU();

	/**
	 *	Construct the collision matrix Qhat by calculating X * Q * X^(-1) and
	 *	Y * Q * Y^(-1) respectively
	 *	@param 	i 	    row index
	 *	@param 	j 	    column index
	 *	@param 	Q 	    collision matrix
	 */
	void Qhat_X(int i, int j, cmat_t& Q) const;
	void Qhat_Y(int i, int j, cmat_t& Q) const;

	/**
	 *	Calculate the macroscopic variables (density, current, velocities)
	 *	@file 	QLBcpu.cpp 
	 */
	void calculate_macroscopic_vars();	
	
	/**
	 *	Calculate the spreads (deltaX and deltaY)
	 *	@file 	QLBcpu.cpp 
	 */
	void calculate_spread();

	/**
	 *	Print the current spreads in the format:
	 *	time  -  deltaX  -  deltaY  -  [exact solution if V=free]
	 *	@file 	QLB.cpp 
	 */	
	void print_spread();

	// === Potential ===
	
	/**
	 *	Harmonic potential: V(r) = g * -1/2 * m * w0^2 * r^2
	 *	with w0 = 1 / (2m * delta0^2)
	 *	@param i        row index
	 *	@param j        column index 
	 */
	float_t V_harmonic(int i, int j) const;
	
	/**
	 *	Free-particle potential: V(r) = 0
	 *	@param i        row index
	 *	@param j        column index 
	 */
	float_t V_free(int i, int j) const;
	
	/**
	 *	Barrier potential: V(r) = g * V0  if i >= L/2
	 *	                        = 0   else
	 *	where V0 is given by  m * L^2 * w0^2 / 8  with w0 = 1 / (2m * delta0^2)
	 *	@param i        row index
	 *	@param j        column index 
	 */
	float_t V_barrier(int i, int j) const;

	/**
	 *	GP potential: V(r) = g * |spinor|^2
	 *	@param i        row index
	 *	@param j        column index 
	 */
	float_t V_GP(int i, int j) const;
	
	/**
	 *	Set the potential array V according to V_indx
	 *	@file QLB.cpp
	 */
	void set_potential_array();

	// === Rendering ===

	/**
	 *	Setup all OpenGL context - this call is mandatory to use QLB::render()
	 *  @param  static_viewer   boolean whether this is used by the static viewer
	 *	@file 	QLBgraphics.cpp
	 */
	void init_GL(bool static_viewer);
	
	/**
	 *	Calculate the vertices by copying the norm of the desired spinor matrix
	 *	to array_vertex_
	 *	@param  tid       thread id in [0, nthreads)
	 *	@param  nthreads  number of threads
	 *	@file QLBgraphics.cpp 
	 */
	void calculate_vertex(int tid, int nthreads);

	/**
	 *	Calculate the vertices by copying the norm of the desired spinor matrix
	 *	or the potential directly to the vertex VBO by a CUDA kernel
	 *	@file QLBcuda.cu 
	 */	
	void calculate_vertex_cuda();
	void calculate_vertex_V_cuda();

	/**
	 *	Calculate the normals depending on array_vertex_ (QLB::calculate_vertex
	 *	should be called prior)
 	 *	@param  tid       thread id in [0, nthreads)
	 *	@param  nthreads  number of threads
	 *	@file QLBgraphics.cpp 
	 */
	void calculate_normal(int tid, int nthreads);
	
	/**
	 *	Calculate the normals of the desired spinor matrix or the potential and
	 *	copy the result directly to the normal VBO by a CUDA kernel (this call 
	 *	can be overlapped with 'calculate_vertex_*_cuda()' calls)
	 *	@file QLBcuda.cu 
	 */	
	void calculate_normal_cuda();
	void calculate_normal_V_cuda();

	/**
	 *	Scale the vertices according to 'scaling_' (This function is used by the
	 *	StaticViewer)
	 *	@param 	change_scaling	-1: decrease by factor of 2.0
	 * 	                         1: increase by factor of 2.0  
	 *	@file QLBgraphics.cpp 
	 */
	void scale_vertex(int change_scaling);
	
	/** 
	 *	Render the current scene 
	 *	@file	QLBgraphics.cpp
	 */
	void render();
	
	/** 
	 *	Render the current scene (used by the static viewer) 
	 *	@param  VBO_changed   boolean whether the VBO's must be updated
	 *	@file	QLBgraphics.cpp
	 */
	void render_statically(bool VBO_changed);
	
	// === IO ===

	/** 
	 *	get_device_arrays 
	 *	@param		
	 */
	void get_device_arrays();
	
	/** 
 	 *	Write the current content of the matrix or vector to STDOUT 
	 *	@param	m	matrix to be printed
	 *	@file 	QLB.cpp
	 */
	void print_matrix(const cmat_t& m) const;
	void print_matrix(const fmat_t& m) const;
		
	/** 
 	 *	Write the current content of the k-th vector element in the 
	 *	matrix (e.g spinor_) to STDOUT 
	 *	@param	m	matrix to be printed
	 *	@param  k 	index of vector in [0,4)
	 *	@file 	QLB.cpp 
	 */
	void print_matrix(const c4mat_t& m, std::size_t k) const;
	
	/** 
	 *	Write the all spreads [0, dt*t_] to 'spread.dat'.
	 *	@file 	QLB.cpp 
	 */
	void write_spread();
	
	/** 
 	 *	Write the current content of all specified quantities (given by 
 	 *	QLBopt's plot_) to the corresponding file(s). 
 	 *	Note: consecutive calls of this function will override the last content
 	 *	      of the files
	 *	@file 	QLB.cpp 
	 */
	void write_content_to_file();
	
	/** 
 	 *	Dump the vertex_array and normal_array to a binary file.
 	 *	@param  static_viewer   boolean whether this command is invoked from the
 	 *	                        static viewer 
	 *	@file 	QLBdump.cpp 
	 */
	void dump_simulation(bool static_viewer);

	/** 
	 *	Adjust the scaling of the rendered scene
	 *	@param 	change_scaling	-1: decrease by factor of 2.0
	 * 	                         1: increase by factor of 2.0
	 *	@file	QLB.cpp 
	 */
	void change_scaling(int change_scaling);
	
	/** 
	 *	Update the constants (d_scaling, d_current_scene) on the device
	 *	@file	QLBcuda.cu 
	 */
	void update_device_constants();
	
	/** 
	 *	Parse a config file given by 'filename' and select the best possible
	 *	block sizes and grid sizes for the system size L_. If the file is 
	 *	default values are being used.
	 *	@param	filename    name of the config file.
	 *	@file	QLBcuda.cu 
	 */
	void set_block_and_grid_size(std::string filename);  
	
	// === Getter ===
	inline unsigned L() const { return L_;  }
	inline float_t dx() const { return dx_; }
	inline float_t mass() const { return mass_; }
	inline float_t t() const { return t_; }
	inline float_t dt() const { return dt_; }
	inline float_t delta0() const { return delta0_; }
	inline int V() const { return V_indx_; }
	inline float_t scaling() const { return scaling_;}
	inline scene_t current_scene() const { return current_scene_; }
	inline render_t current_render() const { return current_render_; }
	inline bool draw_potential() const { return draw_potential_; }
	inline QLBopt opt() const { return opt_; }
	inline QLBparser parser() const { return parser_; }
	
	// === Setter ===
	inline void set_draw_potential(bool dp)     { draw_potential_ = dp; }
	inline void set_current_render(render_t cr) { current_render_ = cr; }
#ifdef QLB_HAS_CUDA
	inline void set_block1(dim3 block1) { block1_ = block1; }
	inline void set_block4(dim3 block4) { block4_ = block4; }
	inline void set_grid1(dim3 grid1) { grid1_ = grid1; }
	inline void set_grid4(dim3 grid4) { grid4_ = grid4; }
#endif
	void set_current_scene(scene_t current_scene);
	
private:
	// === Simulation variables ===
	const unsigned L_;
	float_t dx_;
	float_t mass_;
	
	unsigned t_;
	float_t dt_;

	std::vector<float_t> deltax_;
	std::vector<float_t> deltay_;
	float_t delta0_;
	
	int V_indx_;
	const float_t g_;
	
	barrier_t barrier;
	std::atomic<int> flag_;
	
	// === Arrays CPU === 
	c4mat_t spinor_;
	c4mat_t spinoraux_;
	c4mat_t spinorrot_;
	cmat_t  currentX_;
	cmat_t  currentY_;
	cmat_t  veloX_;
	cmat_t  veloY_;
	cmat_t  wrot_;
	cmat_t  rho_;
	fmat_t  V_;
	
	// === Arrays GPU ===
#ifdef QLB_HAS_CUDA 
	cuFloatComplex* d_spinor_;
	cuFloatComplex* d_spinoraux_;
	cuFloatComplex* d_spinorrot_;
	float* d_V_;
	
	dim3 block1_;
	dim3 block4_;
	dim3 grid1_;
	dim3 grid4_;
	
	float3* d_vertex_ptr_;
	float3* d_normal_ptr_;
#endif 

	// === OpenGL context ===
	bool GL_is_initialzed_;
	scene_t  current_scene_;
	render_t current_render_;
	bool draw_potential_;
	float_t scaling_;
	
	std::vector<unsigned> array_index_solid_;
	std::vector<unsigned> array_index_wire_;  
	std::vector<float>    array_vertex_;
	std::vector<float>    array_normal_;
	
	VBO vbo_vertex;
	VBO vbo_normal;
	VBO vbo_index_solid;
	VBO vbo_index_wire;
	
	// === IO ===
	QLBparser parser_;
	QLBopt opt_;
	std::ofstream fout;
};

#endif /* QLB.hpp */
