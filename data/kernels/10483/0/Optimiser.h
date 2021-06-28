#ifndef OPTIMISER_H
#define OPTIMISER_H

class ExperimentalScenario;
typedef std::shared_ptr<ExperimentalScenario> ExperimentalScenarioPtr;

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class SpeciesRoadPatches;
typedef std::shared_ptr<SpeciesRoadPatches> SpeciesRoadPatchesPtr;

class TrafficProgram;
typedef std::shared_ptr<TrafficProgram> TrafficProgramPtr;

class DesignParameters;
typedef std::shared_ptr<DesignParameters> DesignParametersPtr;

class Vehicle;
typedef std::shared_ptr<Vehicle> VehiclePtr;

class EarthworkCosts;
typedef std::shared_ptr<EarthworkCosts> EarthworkCostsPtr;

class UnitCosts;
typedef std::shared_ptr<UnitCosts> UnitCostsPtr;

class Species;
typedef std::shared_ptr<Species> SpeciesPtr;

class UnitCosts;
typedef std::shared_ptr<UnitCosts> UnitCostSPtr;

class OtherInputs;
typedef std::shared_ptr<OtherInputs> OtherInputsPtr;

class VariableParameters;
typedef std::shared_ptr<VariableParameters> VariableParametersPtr;

class Economic;
typedef std::shared_ptr<Economic> EconomicPtr;

class Traffic;
typedef std::shared_ptr<Traffic> TrafficPtr;

class Region;
typedef std::shared_ptr<Region> RegionPtr;

class ThreadManager;
typedef std::shared_ptr<ThreadManager> ThreadManagerPtr;

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class MonteCarloROV;
typedef std::shared_ptr<MonteCarloROV> MonteCarloROVPtr;

typedef std::shared_ptr<Gnuplot> GnuplotPtr;

/**
 * Class for managing the optimisation process
 */
class Optimiser : public std::enable_shared_from_this<Optimiser> {

public:
    // ENUMERATIONS ///////////////////////////////////////////////////////////
    typedef enum {
        NOPENALTY,      /**< No penalty at all */
        SIMPLEPENALTY,  /**< Penalty when building in certain areas */
        MTE,            /**< Set a minimum population to maintain per species */
        CONTROLLED      /**< Controlled animal population */
    } Type;

    typedef enum {
        ALGO1, /**< Regression Monte Carlo (Tsitsiklis and Van Roy) */
        ALGO2, /**< Regression Monte Carlo (Longstaff and Schwartz) */
        ALGO3, /**< Parametric Control (Guyon and Henry-Labordere) */
        ALGO4, /**< Regression Monte Carlo, State & Control (Kharroubi et al.) */
        ALGO5, /**< Regression Monte Carlo, State, Control & Recomputation */
        ALGO6, /**< Regression Monte Carlo, State, Control, Recomputation & Switching (Langrene et al.) */
        ALGO7  /**< Regression Monte Carlo with targeted end population (Zhang et al.) */
    } ROVType;

    typedef enum {
        COMPUTATION_FAILED,
        COMPUTATION_SUCCESS
    } ComputationStatus;

    typedef enum {
        CUBIC_SPLINE,
        MULTI_LOC_LIN_REG,
    } InterpolationRoutine;

    typedef enum {
        COMPLETE,
        STALLED,
        MAX_GENS
    } StoppingCriterion;

    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs an %Optimiser object with default values.
     */
    Optimiser();

    /**
     * Constructor II
     *
     * Constructs an %Optimiser object with assigned values
     *
     * @param programs as const std::vector<TrafficProgramPtr>&
     * @param oInputs as OtherInputsPtr
     * @param desParams as DesignParametersPtr
     * @param earthworks as EarthworkCostsPtr
     * @param unitCosts as UnitCostsPtr
     * @param varParams as VariableParametersPtr
     * @param species as const std::vector<SpeciesPtr>&
     * @param economic as EconomicPtr
     * @param traffic as TrafficPtr
     * @param region as RegionPtr
     * @param mr as double
     * @param cf as unsigned long
     * @param gens as unsigned long
     * @param sg as unsigned long
     * @param popSize as unsigned long
     * @param stopTol as double
     * @param confInt as double
     * @param confLvl as double
     * @param habGridRes as unsigned long
     * @param solScheme as std::string
     * @param noRuns as unsigned long
     * @param type as Optimiser::Type
     * @param threader as ThreadManagerPtr
     * @param elite as double
     * @param msr as double
     * @param gpu as bool (default = false)
     * @param method as Optimiser:: (default = Optimiser::ALGO5)
     */
    Optimiser(double mr, double cf, unsigned long gens, unsigned long
            popSize, double stopTol, double confInt, double confLvl, unsigned
            long habGridRes, unsigned long surrDimRes, std::string solScheme,
            unsigned long noRuns, Optimiser::Type type, unsigned long sg,
            double msr, unsigned long learnSamples, bool gpu=false,
            Optimiser::ROVType method = Optimiser::ALGO5, InterpolationRoutine
            interp = Optimiser::MULTI_LOC_LIN_REG);
    /**
     * Destructor
     */
    ~Optimiser();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the current ExperimentalScenario
     *
     * @return ExperimentalScenario as ExperimentalScenarioPtr
     * @note Currently this is part of the Optimiser class so we cannot create
     * parallel experiments. If we want to do this, copies of the Optimiser
     * will have to be constructed and passed to different threads where the
     * results are saved to the shared matrix of results. When each experiment
     * is complete, the forked Optimiser can be deleted.
     */
    ExperimentalScenarioPtr getScenario() {
        return this->scenario;
    }
    /**
     * Sets the current ExperimentalScenario
     *
     * @param scenario as ExperimentalScenarioPtr
     */
    void setScenario(ExperimentalScenarioPtr scenario) {
        this->scenario.reset();
        this->scenario = scenario;
    }

    /**
     * Returns the type of optimisation process used
     *
     * @return Ecological attribution method as Optimiser::Type
     */
    Optimiser::Type getType() {
        return this->type;
    }
    /**
     * Sets the type of optimisation process used
     *
     * @param type as Optimiser::Type
     */
    void setType(Optimiser::Type type) {
        this->type = type;
    }

    /**
     * Returns the current GA population of roads. These roads are only defined
     * by their intersection points.
     *
     * All roads have the same start and end points (in three dimensions) but
     * the optimisation algorithm only modifies the intervening points.
     *
     * @return Roads as const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd& getCurrentRoads() {
        return this->currentRoadPopulation;
    }
    /**
     * Sets the current GA population of roads
     *
     * @param roads as const Eigen::MatrixXd&
     */
    void setCurrentRoads(const Eigen::MatrixXd& roads) {
        this->currentRoadPopulation = roads;
    }

    /**
     * Returns the best roads computed by the optimiser.
     *
     * @return Roads as const std::vector< std::vector<RoadPtr> >&
     */
    const std::vector< std::vector<RoadPtr> >& getBestRoads() {
        return this->bestRoads;
    }
    /**
     * Sets the best roads computed by the optimiser
     *
     * @param roads as const std::vector< std::vector<RoadPtr> >&
     */
    void setBestRoads(const std::vector< std::vector<RoadPtr> >& roads) {
        this->bestRoads = roads;
    }

    /**
     * Returns the different switching programs
     *
     * @return Program as const std::vector<TrafficProgramPtr>&
     */
    const std::vector<TrafficProgramPtr>& getPrograms() {
        return this->programs;
    }
    /**
     * Sets the different switching programs
     *
     * @param programs as const std::vector<TrafficProgramPtr>&
     */
    void setPrograms(const std::vector<TrafficProgramPtr>& program) {
        this->programs = program;
    }

    /**
     * Returns the miscellaneous inputs needed by the software
     *
     * @return Inputs as OtherInputsPtr
     */
    OtherInputsPtr getOtherInputs() {
        return this->otherInputs;
    }
    /**
     * Sets the miscellaneous inputs needed by the software
     *
     * @param inputs as OtherInputsPtr
     */
    void setOtherInputs(OtherInputsPtr inputs) {
        this->otherInputs.reset();
        this->otherInputs = inputs;
    }

    /**
     * Returns the road design parameters
     *
     * @return Road design parameters as DesignParamsPtr
     */
    DesignParametersPtr getDesignParameters() {
        return this->designParams;
    }
    /**
     * Sets the road design parameters
     *
     * @param params as DesignParamsPtr
     */
    void setDesignParams(DesignParametersPtr params) {
        this->designParams.reset();
        this->designParams = params;
    }

    /**
     * Returns the earthwork requirements used
     *
     * @return Earthworks as EarthworkCostsPtr
     */
    EarthworkCostsPtr getEarthworkCosts() {
        return this->earthworks;
    }
    /**
     * Sets the earthwork requirements used
     *
     * @param earthworks as EarthworkCostsPtr
     */
    void setEarthworkCosts(EarthworkCostsPtr earthworks) {
        this->earthworks.reset();
        this->earthworks = earthworks;
    }

    /**
     * Returns the unit costs
     *
     * @return Unit costs as UnitCostsPtr
     */
    UnitCostsPtr getUnitCosts() {
        return this->unitCosts;
    }
    /**
     * Sets the unit costs
     *
     * @param costs as UnitCostsPtr
     */
    void setUnitCosts(UnitCostsPtr costs) {
        this->unitCosts.reset();
        this->unitCosts = costs;
    }

    /**
     * Returns the current generation in the optimisation process
     *
     * @return Current generation as double
     */
    double getGeneration() {
        return this->generation;
    }
    /**
     * Sets the current generation in the optimisation process
     *
     * @param gen as double
     */
    void setGeneration(double gen) {
        this->generation = gen;
    }

    /**
     * Returns the variable parameters
     *
     * @return Variable parameters as VariableParametersPtr
     */
    VariableParametersPtr getVariableParams() {
        return this->variableParams;
    }
    /**
     * Sets the variable parameters
     *
     * @param varParams as VariableParametersPtr
     */
    void setVariableParams(VariableParametersPtr varParams) {
        this->variableParams.reset();
        this->variableParams = varParams;
    }

    /**
     * Returns the species details
     *
     * @return Species as const std::vector<SpeciesPtr>&
     */
    const std::vector<SpeciesPtr>& getSpecies() {
        return this->species;
    }
    /**
     * Sets the species details
     *
     * @param species as const std::vector<SpeciesPtr>&
     */
    void setSpecies(const std::vector<SpeciesPtr>& species) {
        this->species = species;
    }

    /**
     * Returns the economic data
     *
     * @return Economic data as EconomicPtr
     */
    EconomicPtr getEconomic() {
        return this->economic;
    }
    /**
     * Sets the economic data
     *
     * @param econ as EconomicPtr
     */
    void setEconomic(EconomicPtr econ) {
        this->economic.reset();
        this->economic = econ;
    }

    /**
     * Returns the traffic data
     *
     * @return Traffic data as TrafficPtr
     */
    TrafficPtr getTraffic() {
        return this->traffic;
    }
    /**
     * Sets the traffic data
     *
     * @param traffic as TrafficPtr
     */
    void setTraffic(TrafficPtr traffic) {
        this->traffic.reset();
        this->traffic = traffic;
    }

//    /**
//     * Returns the TrafficProgram over which the optimisation is run.
//     *
//     * @return TrafficProgram as TrafficProgramPtr
//     */
//    TrafficProgramPtr getTrafficProgram() {
//        return this->trafficProgram;
//    }
//    /**
//     * Sets the TrafficProgram over which the optimisation is run.
//     *
//     * @param tp as TrafficProgramPtr
//     */
//    void setTrafficProgram(TrafficProgramPtr tp) {
//        this->trafficProgram.reset();
//        this->trafficProgram = tp;
//    }

    /**
     * Returns the region data
     *
     * @return Region data as RegionPtr
     */
    RegionPtr getRegion() {
        return this->region;
    }
    /**
     * Sets the region data
     *
     * @param region as RegionPtr
     */
    void setRegion(RegionPtr region) {
        this->region.reset();
        this->region = region;
    }

    /**
     * Returns the GA mutation rate
     *
     * @return Mutation rate as double
     */
    double getMutationRate() {
        return this->mutationRate;
    }
    /**
     * Sets the GA mutation rate
     *
     * @param rate as double
     */
    void setMutationRate(double rate) {
        this->mutationRate = rate;
    }

    /**
     * Returns the number of optimisation runs to perform for population-based
     * optimisation.
     *
     * @return Number of runs as unsigned long
     */
    unsigned long getNoRuns() {
        return this->noRuns;
    }
    /**
     * Sets the number of optimisation runs to perform.
     *
     * @param noRuns as unsigned long
     */
    void setNoRuns(unsigned long noRuns) {
        this->noRuns = noRuns;
    }

    /**
     * Returns the maximum number of desired samples for building the surrogate
     *
     * @return Maximum learning samples as unsigned long
     */
    unsigned long getLearnSamples() {
        return this->learnSamples;
    }
    /**
     * Sets the maximum number of desired samples for building the surrogate
     *
     * @param ls as unsigned long
     */
    void setLearnSamples(unsigned long ls) {
        this->learnSamples = ls;
    }

    /**
     * Returns the GA crossover fraction
     *
     * @return Crossover fraction as double
     */
    double getCrossoverFrac() {
        return this->crossoverFrac;
    }
    /**
     * Sets the GA crossover fraction
     *
     * @param frac as double
     */
    void setCrossoverFrac(double frac) {
        this->crossoverFrac = frac;
    }

    /**
     * Returns the max number of GA generations
     *
     * @return Generations as unsigned long
     */
    unsigned long getMaxGens() {
        return this->generations;
    }
    /**
     * Sets the max number of GA generations
     *
     * @params gens as unsigned long
     */
    void setMaxGens(unsigned long gens) {
        this->generations = gens;
    }

    /**
     * Returns the max number of generations to continue without improvement
     *
     * @return Number of stall generations permitted as unsigned long
     */
    unsigned long getStallGens() {
        return this->stallGenerations;
    }
    /**
     * Sets the max number of generations to continue without improvement
     *
     * @param sg as unsigned long
     */
    void setStallGens(unsigned long sg) {
        this->stallGenerations = sg;
    }

    /**
     * Returns the GA population size
     *
     * @return Population size as unsigned long
     */
    unsigned long getGAPopSize() {
        return this->populationSizeGA;
    }
    /**
     * Sets the GA population size
     *
     * @param size as unsigned long
     */
    void setPopSize(unsigned long size) {
        this->populationSizeGA = size;
    }

    /**
     * Returns the optimiser stopping tolerance
     *
     * @return Stopping tolerance as double
     */
    double getStoppingTol() {
        return this->stoppingTol;
    }
    /**
     * Sets the optimiser stopping tolerance
     *
     * @param tol as double
     */
    void setStoppingTol(double tol) {
        this->stoppingTol = tol;
    }

    /**
     * Returns the confidence interval used for simulation results.
     *
     * This number (X) has different uses depending on the simulation model:
     * 1.   Constant full traffic flow - X is used to compute the end animal
     *      population that the road has a 95% chance of exceeding.
     * 2.   Stochastic dynamic programming - X is the probability that using
     *      control Y will result in the population exceeding the threshold.
     *
     * Alternately, X can be interpreted as the proportion of roads exceeding
     * the number we use to define the road (end population, profit, etc.)
     * when comparing it to other roads. That means that X is multi-purpose.
     * @return Confidence interval as double
     */
    double getConfidenceInterval() {
        return this->confInt;
    }
    /**
     * Sets the confidence interval used for simulation results.
     *
     * @param confidence as double
     */
    void setConfidence(double confidence) {
        this->confInt = confidence;
    }

    /**
     * Returns the confidence level used for simulations/sampling
     *
     * This is also used when computing required sample sizes.
     * @return Confidence level of results as double
     */
    double getConfidenceLevel() {
        return this->confLvl;
    }
    /**
     * Sets the confidence level used for simulations/sampling
     *
     * @param confidence as double
     */
    void setConfidenceLevel(double confidence) {
        this->confLvl = confidence;
    }

    /**
     * Returns the habitat grid resolution
     *
     * @return Grid resolution as unsigned long
     */
    unsigned long getGridRes() {
        return this->habGridRes;
    }
    /**
     * Sets the habitat grid resolution
     *
     * @param res as double
     */
    void setGridRes(unsigned long res) {
        this->habGridRes = res;
    }

    /**
     * Returns the surrogate resolution in each dimension
     *
     * @return Resolution as unsigned long
     */
    unsigned long getSurrDimRes() {
        return this->surrDimRes;
    }
    /**
     * Sets the surrogate resolution in each dimension
     *
     * @param res as unsigned long
     */
    void setSurrDimRes(unsigned long res) {
        this->surrDimRes = res;
    }

    /**
     * Returns the solution scheme of the solver
     *
     * @return Solution scheme as std::string
     */
    std::string getSolutionScheme() {
        return this->solutionScheme;
    }
    /**
     * Sets the solution scheme of the solver
     *
     * @param scheme as std::string
     */
    void setSolutionScheme(std::string scheme) {
        this->solutionScheme = scheme;
    }

    /**
     * Returns the Thread Manager
     *
     * @return ThreadManager as ThreadManagerPtr
     */
    ThreadManagerPtr getThreadManager() {
        return this->threader;
    }
    /**
     * Sets the ThreadManager
     *
     * @param threader as ThreadManagerPtr
     */
    void setThreadManager(ThreadManagerPtr threader) {
        this->threader.reset();
        this->threader = threader;
    }

    /**
     * Returns the thread manager for the GPUs
     *
     * @return threader as ThreadManagerPtr
     */
    ThreadManagerPtr getThreadManagerGPU() {
        return this->threaderGPU;
    }
    /**
     * Sets the thread manager for GPUs
     *
     * @param threader as ThreadManagerPtr
     */
    void setThreadManagerGPU(ThreadManagerPtr threader) {
        this->threaderGPU.reset();
        this->threaderGPU = threader;
    }

    /**
     * Returns the proportion of the population to retain as elite
     *
     * @return Elite individuals proportion as double
     */
    double getEliteIndividuals() {
        return this->eliteIndividuals;
    }
    /**
     * Sets the proportion of the population to retain as elite
     *
     * @param e as double
     */
    void setEliteIndividuals(double e) {
        this->eliteIndividuals = e;
    }

    /**
     * Returns the maximum rate at which to extract samples from a generation
     *
     * Returns the maximum rate at which to extract samples from a generation
     * so as to determine the surrogate model for improving the performance of
     * the GA.
     *
     * @return Max sample rate as double
     */
    double getMaxSampleRate() {
        return this->maxSampleRate;
    }
    /**
     * Sets the sampling rate for building surrogate models
     *
     * @param msr as double
     */
    void setMaxSampleRate(double msr) {
        this->maxSampleRate = msr;
    }

    /**
     * Returns the handle to the output interface for Gnuplot (standard)
     *
     * @return Handle as GnuplotPtr
     */
    GnuplotPtr getPlotHandle() {
        return this->plothandle;
    }
    /**
     * Sets the handle for the terrain plots
     *
     * @param ph as GnuplotPtr
     */
    void setPlotHandle(GnuplotPtr ph) {
        this->plothandle.reset();
        this->plothandle = ph;
    }

    /**
     * Returns the handle to the output interface for Gnuplot (surrogates)
     *
     * @return Handle as GnuplotPtr
     */
    GnuplotPtr getSurrPlotHandle() {
        return this->surrPlotHandle;
    }
    /**
     * Sets the handle for the surrogate models
     *
     * @param ph as GnuplotPtr
     */
    void setSurrPlotHandle(GnuplotPtr ph) {
        this->surrPlotHandle.reset();
        this->surrPlotHandle = ph;
    }

    /**
     * Returns a Boolean of whether we are using a GPU or not
     *
     * @return Whether we are using a GPU as bool
     */
    bool getGPU() {
        return this->gpu;
    }
    /**
     * Sets a Boolean of whether we are using a GPU or not
     *
     * @param gpu as bool
     */
    void setGPU(bool gpu) {
        this->gpu = gpu;
    }

    /**
     * Returns the ROV method used in the optimisation
     *
     * @return Method as Optimiser::ROVType
     */
    Optimiser::ROVType getROVMethod() {
        return this->method;
    }
    /**
     * Sets the ROV method used in the optimisation
     *
     * @param method as Optimiser::ROVType
     */
    void setROVMethod(Optimiser::ROVType method) {
        this->method = method;
    }

    /**
     * Returns the type of interpolation routine used in the surrogate models
     *
     * @return Interpolation method as Optimiser::InterpolationRoutine
     */
    Optimiser::InterpolationRoutine getInterpolationRoutine() {
        return this->interp;
    }
    /**
     * Sets the type of interpolation routine used in the surrogate models
     *
     * @param ir as Optimiser::InterpolationRoutine
     */
    void setInterpolationRoutine(Optimiser::InterpolationRoutine ir) {
        this->interp = ir;
    }

    /**
     * Returns the relative path of the root folder
     *
     * @return Root folder path (relative to program root) as std::string
     */
    const std::string getRootFolder() {
        return this->rootFolder;
    }
    /**
     * Sets the relative path of the root folder
     *
     * @param rf as std::string
     */
    void setRootFolder(std::string rf) {
        this->rootFolder = rf;
    }

    /**
     * Returns the X values file path
     *
     * @return X values file as std::string
     */
    const std::string getXValuesFile() {
        return this->xValuesFile;
    }
    /**
     * Sets the X values file path
     *
     * @param xvf as std::string
     */
    void setXValuesFile(std::string xvf) {
        this->xValuesFile = xvf;
    }

    /**
     * Returns the Y values file path
     *
     * @return Y values file path as std::string
     */
    const std::string getYValuesFile() {
        return this->yValuesFile;
    }
    /**
     * Sets the Y values file path
     *
     * @param yvf as std::string
     */
    void setYValuesFile(std::string yvf) {
        this->yValuesFile = yvf;
    }

    /**
     * Returns the Z values file path
     *
     * @return Z values file path as std::string
     */
    const std::string getZValuesFile() {
        return this->zValuesFile;
    }
    /**
     * Sets the Z values file path
     *
     * @param zvf as std::string
     */
    void setZValuesFile(std::string zvf) {
        this->zValuesFile = zvf;
    }

    /**
     * Returns the vegetation data file path
     *
     * @return Vegetation data file path as std::string
     */
    const std::string getVegetationFile() {
        return this->vegetationFile;
    }
    /**
     * Sets the vegetation data file path
     *
     * @param vf as std::string
     */
    void setVegetationFile(std::string vf) {
        this->vegetationFile = vf;
    }

    /**
     * Returns the acquisition data file path
     *
     * @return Acquisition data file path as std::string
     */
    const std::string getAcquisitionFile() {
        return this->acquisitionFile;
    }
    /**
     * Sets the acquisition data file path
     *
     * @param af as std::string
     */
    void setAcquisitionFile(std::string af) {
        this->acquisitionFile = af;
    }

    /**
     * Returns the soil data file path
     *
     * @return Soil data file path as std::string
     */
    const std::string getSoilFile() {
        return this->soilFile;
    }
    /**
     * Sets the soil data file path
     *
     * @param sf as std::string
     */
    void setSoilFile(std::string sf) {
        this->soilFile = sf;
    }

    /**
     * Returns the commodities data files
     *
     * @return Commodities data files as std::vector<std::string>
     */
    const std::vector<std::string> getCommoditiesFiles() {
        return this->commoditiesFiles;
    }
    /**
     * Sets the commodities data files
     *
     * @param cf as std::vector<std::string>
     */
    void setCommoditiesFiles(std::vector<std::string>& cf) {
        this->commoditiesFiles = cf;
    }

    /**
     * Returns the fuels data files
     *
     * @return Fuels data files as std::vector<std::string>&
     */
    const std::vector<std::string> getFuelsFiles() {
        return this->fuelsFiles;
    }
    /**
     * Sets the fuels data files
     *
     * @param ff as std::vector<std::string>&
     */
    void setFuelsFiles(std::vector<std::string>& ff) {
        this->fuelsFiles = ff;
    }

    /**
     * Returns the vehicles data files
     *
     * @return Vehicles data files as std::vector<std::string>
     */
    const std::vector<std::string> getVehiclesFiles() {
        return this->vehiclesFiles;
    }
    /**
     * Sets the vehicles data files
     *
     * @param vf as std::vector<std::string>&
     */
    void setVehiclesFiles(std::vector<std::string>& vf) {
        this->vehiclesFiles = vf;
    }

    /**
     * Returns the species data files
     *
     * @return Species data files as std::vector<std::string>
     */
    const std::vector<std::string> getSpeciesFiles() {
        return this->speciesFiles;
    }
    /**
     * Sets the species data files
     *
     * @param sf as std::vector<std::string>&
     */
    void setSpeciesFiles(std::vector<std::string>& sf) {
        this->speciesFiles = sf;
    }

    /**
     * Returns the comparison road for ROV profit computation (if exists)
     *
     * @return Comparison road as RoadPtr
     */
    RoadPtr getComparisonRoad() {
        return this->comparisonRoad;
    }
    /**
     * Sets the comparison road for ROV profit computation
     *
     * If there is a comparison road, the algorithm does not take into account
     * commodity sales revenue as it is assumed to continue due to traffic
     * being diverted to the alternate road. Instead, the difference in unit
     * traffic along this road and the candidate road being evaluated is the
     * unit profit for the candidate road. If no comparison road is provided,
     * it is assumed that none exists and the reduction in traffic (and
     * therefore sales revenue) is lost completely.
     *
     * @param cr as RoadPtr
     */
    void setComparisonRoad(RoadPtr cr) {
        this->comparisonRoad.reset();
        this->comparisonRoad = cr;
    }

    /**
     * Returns the maximum possible benefit of using an alternative road
     *
     * @return Max benefit as double
     */
    double getMaxROVBenefit() {
        return this->maxROVBenefit;
    }
    /**
     * Sets the maximum possible benefit of using an alternative road
     *
     * @param benefit as double
     */
    void setMaxROVBenefit(double benefit) {
        this->maxROVBenefit = benefit;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Initialises the experimental results storage for retaining data between
     * runs
     *
     * @note This function must be called once all object attributes such as
     * VarParams have been assigned non-null pointers
     */
    virtual void initialiseExperimentStorage();

    /**
     * Initialises the storage containers in the object (bestRoads etc.)
     *
     * @note This function must be called once all object attributes such as
     * VarParams have been assigned non-null pointers
     */
    virtual void initialiseStorage();

    /**
     * Runs the algorithm to optimise the road with the possibility to enter
     * the animal habitat zones. Results are saved to output files. If the
     * argument "true" is passed to the function, the following plots are
     * generated as the optimisation proceeds. Otherwise, no plots are
     * generated:
     * 1. 3D plot of best road on terrain
     * 2. 2D plot of best road with respect to habitat
     * 3. Surrogate function
     * 4. Best results history
     *
     * This routine contains code to simply plot the result. The actual
     * optimisation procedures are contained in derived classes.
     *
     * @param plot as bool
     */
    virtual void optimise(bool plot = false);

    /**
     * Computes the habitat map for every species
     */
    void computeHabitatMaps();

    /**
     * Computes the expected present value of a constant unit of use of every
     * uncertain parameter over the design horizon.
     */
    void computeExpPv();

    /**
     * Returns a reference to the surrogate function so that it may be used
     *
     * @return Surrogate function as std::vector<std::vector<std::vector<alglib::spline1dinterpolant>>>&
     */
    std::vector<std::vector<std::vector<alglib::spline1dinterpolant>>>& getSurrogate() {
        return this->surrogate;
    }
    /**
     * Sets the surrogate function so that it may be called at a later stage
     *
     * The surrogate function accepts a pointer to a road object to compute its
     * operating cost (based on a learned function), which is returned as a
     * double.
     *
     * @param surrogate as std::vector<std::vector<std::vector<alglib::spline1dinterpolant>>>&
     */
    void setSurrogate(std::vector<std::vector<std::vector<
            alglib::spline1dinterpolant>>>& surrogate) {
        this->surrogate = surrogate;
    }

    /**
     * Returns a reference to the surrogate functions so that they may be used
     *
     * The surrogate model is stored as a fixed-resolution map of predictor
     * points (res*noDims) with corresponding regressed values (res^noDims)
     *
     * The first res x noDims values are the fixed predictor points that can be
     * interpolated. They are sorted by each dimension so that all the 'res'
     * values along each dimension are listed first followed by the 'res'
     * values for the next dimension, and so on. The next 'res*noDims' values
     * are the corresponding predicted values. The way to index is as follows:
     *
     * Each the of these surrogate models for each run are stored in a
     * std::vector where each element is an individual surrogate model. For
     * MTE, there are N 1D surrogates, one for each species. For ROV, there is
     * but one surrogate of (N+1) dimensions, where N dimensions relate to the
     * adjusted population of the species and one relates to the unit profit.
     *
     * 1. Value of predictor ii in dimension jj
     *
     *    double value = this->surrogateML[jj*res + ii]
     *
     * 2. Value of corresponding regressed value for coordinate I[jj_0, jj_1,
     *    ..., jj_(noDims-1)]
     *
     *    int idx = res*noDims;
     *
     *    for (int ii = 0; ii < noDims; ii++) {
     *       idx += I[ii]*res^(noDims - ii - 1);
     *    }
     *
     *    double value = this->surrogateML[idx];
     *
     * @return Surrogates as std::vector<std::vector<std::vector<Eigen::VectorXd>>>&
     */
    std::vector<std::vector<std::vector<Eigen::VectorXd>>>& getSurrogateML() {
        return this->surrogateML;
    }
    /**
     * Sets the surrogate function so that it may be called at a later stage
     *
     * @param std::vector<std::vector<std::vector<Eigen::VectorXd>>>&
     */
    void setSurrogateML(std::vector<std::vector<std::vector<Eigen::VectorXd>>>&
            surr) {
        this->surrogateML = surr;
    }

    /**
     * Evaluates the surrogate model for a given road in the fixed traffic flow
     * case.
     *
     * For each species, a cubic spline of the BSpline class is evaluated
     *
     * @param (input) road as RoadPtr
     * @param (output) pops as Eigen::VectorXd&
     * @param (output) popsSD as Eigen::VectorXd&
     */
    void evaluateSurrogateModelMTE(RoadPtr road, Eigen::VectorXd &pops,
            Eigen::VectorXd &popsSD);

    /**
     * Evaluates the surrogate model for a given road in the fixed traffic flow
     * case using a multiple local linear regression surrogate model.
     *
     * For each species, a multiple local linear regression is evaluated.
     *
     * @param (input) road as RoadPtr
     * @param (output) pops as Eigen::VectorXd&
     * @param (output) popsSD as Eigen::VectorXd&
     */
    void evaluateSurrogateModelMTEML(RoadPtr road, Eigen::VectorXd &pops,
            Eigen::VectorXd &popsSD);
    /**
     * Evaluates the surrogate model for a given road in the fixed traffic flow
     * case.
     *
     * @param (input) road as RoadPtr
     * @param (output) value as double&
     * @param (output) valueSD as double&
     */
    void evaluateSurrogateModelROVCR(RoadPtr road, double &value, double
            &valueSD);

    /**
     * Initialises the program with input data
     *
     * @param inputfile as std::string
     */
    void initialiseFromTextInput(std::string inputfile);

    /**
     * Saves the results for an experimental scenario
     */
    void saveExperimentalResults();

    /**
     * Saves the values for the total population at every generation
     */
    void saveRunPopulation();

    /**
     * Saves the best road data by recomputing all important aspects and saving
     * the data to an external source (on disk).
     *
     * @note This function is used to save all best road data as text. A
     * routine to save the data in binary format is currently in development.
     */
    void saveBestRoadResults();

    /**
     * Computes the results of the best road for reporting and visualisation
     * purposes.
     */
    void computeBestRoadResults();

///////////////////////////////////////////////////////////////////////////////
protected:
    std::vector<std::vector<std::vector<alglib::spline1dinterpolant>>> surrogate;   /**< MTE Surrogate model for evaluating road (for each run) stored as a collection of splines (To be deprecated) */
    std::vector<std::vector<std::vector<Eigen::VectorXd>>> surrogateML;             /**< Surrogate models based on multiple local linear regression */
    RoadPtr comparisonRoad;                                                         /**< Alternative road for ROV */
    double maxROVBenefit;                                                           /**< Logical ceiling for the ROV value */
    ExperimentalScenarioPtr scenario;                                               /**< Current experiment */
    Optimiser::Type type;                                                           /**< Type of ecological incorporation */
    Optimiser::StoppingCriterion stop;                                              /**< Reason for ending optimisation process */
    Eigen::MatrixXd currentRoadPopulation;                                          /**< Current encoded population of roads */
    std::vector< std::vector<RoadPtr> > bestRoads;                                  /**< Best roads */
    std::vector<TrafficProgramPtr> programs;                                        /**< Operational programs */
    OtherInputsPtr otherInputs;                                                     /**< Other inputs */
    DesignParametersPtr designParams;                                               /**< Design parameters */
    EarthworkCostsPtr earthworks;                                                   /**< Earthwork requirements */
    VariableParametersPtr variableParams;                                           /**< Parameters to vary */
    std::vector<SpeciesPtr> species;                                                /**< Species studied */
    EconomicPtr economic;                                                           /**< Economic parameters */
    TrafficPtr traffic;                                                             /**< Traffic details */
    //TrafficProgramPtr trafficProgram;                                             /**< Traffic program used */
    RegionPtr region;                                                               /**< Region of interest */
    UnitCostsPtr unitCosts;                                                         /**< Unit Costs */
    unsigned long generation;                                                       /**< Current generation in optimisation process */
    unsigned long stallGen;                                                         /**< Number of sequential stall generations so far */
    double mutationRate;                                                            /**< Mutation rate */
    double crossoverFrac;                                                           /**< Crossover fraction */
    unsigned long generations;                                                      /**< Generations required */
    unsigned long stallGenerations;                                                 /**< Maximum number of stall generations at which to stop the algorithm */
    unsigned long populationSizeGA;                                                 /**< Population size for GA */
    double stoppingTol;                                                             /**< Stopping tolerance */
    double confInt;                                                                 /**< Required confidence interval */
    double confLvl;                                                                 /**< Desired confidence level (default = 95%) */
    unsigned long habGridRes;                                                       /**< Habitat grid 1D resolution */
    unsigned long surrDimRes;                                                       /**< Resolution along each dimension of the surrogate */
    unsigned long noRuns;                                                           /**< Number of runs to perform */
    unsigned long learnSamples;                                                     /**< Maximum number of samples required by user */
    double eliteIndividuals;                                                        /**< Proportion of elite individuals to retain each generation */
    double maxSampleRate;                                                           /**< Maximum rate at which to perform sampling for surrogate models */
    std::string solutionScheme;                                                     /**< Solution scheme used (i.e. name of experiment) */
    ThreadManagerPtr threader;                                                      /**< Thread manager used for multithreading computations */
    ThreadManagerPtr threaderGPU;                                                   /**< The thread pool for parallel calls to GPUs cannot exceed the number of GPUs */
    OptimiserPtr me();                                                              /**< Creates a shared pointer from this */
    GnuplotPtr plothandle;                                                          /**< Pipe for plotting results */
    GnuplotPtr surrPlotHandle;                                                      /**< Pipe for plotting surrogate model results */
    bool gpu;                                                                       /**< If we are using GPUs to assist computing */
    int gpus;                                                                       /**< The number of GPUs available to use */
    Optimiser::ROVType method;                                                      /**< ROV algorithm */
    Optimiser::InterpolationRoutine interp;                                         /**< Interpolation routine (only used for MTE) */
    std::string rootFolder;                                                         /**< Root folder of the experiments (Input file location) */
    std::string xValuesFile;                                                        /**< X values file used for terrain */
    std::string yValuesFile;                                                        /**< Y values file used for terrain */
    std::string zValuesFile;                                                        /**< Z values file used for terrain */
    std::string vegetationFile;                                                     /**< Vegetation data file */
    std::string acquisitionFile;                                                    /**< Acquisition costs file */
    std::string soilFile;                                                           /**< Soil stabilisation data file */
    std::vector<std::string> commoditiesFiles;                                      /**< Commodities data files */
    std::vector<std::string> fuelsFiles;                                            /**< Fuels data files */
    std::vector<std::string> vehiclesFiles;                                         /**< Vehicles data files */
    std::vector<std::string> speciesFiles;                                          /**< Species data files */

    // Sharing of derived from base
    template <typename Derived>
    std::shared_ptr<Derived> meDerived()
    {
        return std::static_pointer_cast<Derived>(shared_from_this());
    }
};

#endif
