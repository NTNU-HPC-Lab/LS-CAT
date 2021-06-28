/** include file for standard system include files,
 * AND project specific include files that are used frequently, but
 * are changed infrequently */

// If wish to compile on a Windows machine, uncomment the line below
//#include "targetver.h"

#ifndef TRANSPORTBASE_H
#define TRANSPORTBASE_H

#define _USE_MATH_DEFINES
#define NOMINMAX

// SYSTEM-WIDE HEADERS ////////////////////////////////////////////////////////
#include <list>
#include <limits>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <math.h>
#include <float.h>
#include <string>
#include <map>
#include <sstream>
#include <random>
#include <fstream>
#include <regex>
#include <thread>
#include <random>
#include <assert.h>
#include <chrono>
#include <stdexcept>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

// HEADER-ONLY LIBRARIES //////////////////////////////////////////////////////
// EIGEN
#include <Eigen/Dense>
#include <Eigen/Sparse>
// LIBIGL
#include <igl/ceil.h>
#include <igl/colon.h>
#include <igl/cumsum.h>
#include <igl/floor.h>
#include <igl/repmat.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/sort.h>
#include <igl/sortrows.h>
#include <igl/unique.h>
#include <igl/find.h>
#include <igl/mat_max.h>
#include <igl/mat_min.h>
// BOOST

// CTPL
#include <CTPL-master/ctpl_stl.h>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// GNUPLOTIOSTREAM
// Only include in the specific CPP source files where required. The wrapper
// was written with bad practices including initialising static members in the
// header instead of the source file.
#include <gnuplot-iostream-master/gnuplot-iostream.h>

// EXTERNAL PROJECTS INCLUDED IN SOURCE CODE (AS PER USAGE INSTRUCTIONS) //////
#include "utilities/labelmethod.h"
#include "utilities/r8lib.hpp"
#include "utilities/pwl_interp_2d.hpp"
#include "utilities/interpolation.h"

// PROJECT-SPECIFIC HEADERS (MUST MAINTAIN ORDER SHOWN HERE ) /////////////////
#include "main/Optimiser.h"
#include "main/ExperimentalScenario.h"
#include "information/UnitCosts.h"
#include "information/DesignParameters.h"
#include "information/VariableParameters.h"
#include "valuation/Program.h"
#include "information/TrafficProgram.h"
#include "information/OtherInputs.h"
#include "valuation/Uncertainty.h"
#include "ecology/Species.h"
#include "information/Region.h"
#include "information/EarthworkCosts.h"
#include "information/Vehicle.h"
#include "valuation/PolicyMap.h"
#include "valuation/PolicyMapYear.h"
#include "valuation/PolicyMapFrontier.h"
#include "valuation/MonteCarloROV.h"
#include "main/Simulator.h"
#include "information/Economic.h"
#include "information/Traffic.h"
#include "information/Commodity.h"
#include "information/CommodityCovariance.h"
#include "valuation/State.h"
#include "road/HorizontalAlignment.h"
#include "road/VerticalAlignment.h"
#include "road/RoadSegments.h"
#include "road/RoadCells.h"
#include "road/Attributes.h"
#include "road/Costs.h"
#include "road/Road.h"
#include "ecology/HabitatType.h"
#include "ecology/HabitatPatch.h"
#include "utilities/Utility.h"
#include "road/SpeciesRoadPatches.h"
#include "main/ThreadManager.h"
#include "gpuCode/SimulateGPU.h"
#include "gpuCode/knn_cuda_with_indexes.h"
#include "main/RoadGA.h"

// BOOST
#include <boost/filesystem.hpp>
/*
#include <gnuplot-iostream-master/gnuplot-iostream.h>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

#include <boost/filesystem/path.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/array.hpp>
*/

// COMPILED LIBRARIES /////////////////////////////////////////////////////////
//<NONE>

// Introduce new namespaces
//namespace bm = boost::math;

// TODO: reference additional headers your program requires here

// Random number generator seed for all utilised distributions:
static std::default_random_engine generator;

// Global variables
// Max precision for float numbers
static float DBL_PREC = 1e-6;
// Max number of CUDA threads per block (default)
static int maxThreadsPerBlock = 256;
static int maxMultiProcessors = 6;

#endif
