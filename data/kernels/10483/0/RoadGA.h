#ifndef ROADGA_H
#define ROADGA_H

class OtherInputs;
typedef std::shared_ptr<OtherInputs> OtherInputsPtr;

class TrafficProgram;
typedef std::shared_ptr<TrafficProgram> TrafficProgramPtr;

class DesignParameters;
typedef std::shared_ptr<DesignParameters> DesignParametersPtr;

class EarthworkCosts;
typedef std::shared_ptr<EarthworkCosts> EarthworkCostsPtr;

class UnitCosts;
typedef std::shared_ptr<UnitCosts> UnitCostsPtr;

class VariableParameters;
typedef std::shared_ptr<VariableParameters> VariableParametersPtr;

class Species;
typedef std::shared_ptr<Species> SpeciesPtr;

class Economic;
typedef std::shared_ptr<Economic> EconomicPtr;

class Traffic;
typedef std::shared_ptr<Traffic> TrafficPtr;

class Region;
typedef std::shared_ptr<Region> RegionPtr;

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class RoadGA;
typedef std::shared_ptr<RoadGA> RoadGAPtr;

class Road;
typedef std::shared_ptr<Road> RoadPtr;

/**
 * Class for managing the optimal road design process using Genetic Algorithms
 *
 * @note Based on the road design procedure of Jong and Schonfeld (2003)
 */
class RoadGA : public Optimiser {

public:
    // ENUMERATIONS ///////////////////////////////////////////////////////////
    typedef enum {
        STOCHASTIC_UNIFORM,
        REMAINDER,
        UNIFORM,
        ROULETTE,
        TOURNAMENT
    } Selection;

    typedef enum {
        RANK,
        PROPORTIONAL,
        TOP,
        SHIFT,
    } Scaling;

    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a RoadGA object with default values.
     *
     * @note This is the constructor that will typically be used.
     */
    RoadGA();

    /**
     * Constructor II
     *
     * Constructs a RoadGA object with assigned values
     */
    RoadGA(double mr, double cf, unsigned long gens, unsigned long
            popSize, double stopTol, double confInt, double confLvl, unsigned
            long habGridRes, unsigned long surrDimRes, std::string solScheme,
            unsigned long noRuns, Optimiser::Type type, double scale, unsigned
            long learnPeriod, double surrThresh, unsigned long maxLearnNo,
            unsigned long minLearnNo, unsigned long sg, RoadGA::Selection
            selector, RoadGA::Scaling fitscale, double topProp, double
            maxSurvivalRate, int ts, double msr, unsigned long learnSamples,
            bool gpu = false,
            Optimiser::ROVType rovType = Optimiser::ALGO5,
            Optimiser::InterpolationRoutine interp =
            Optimiser::MULTI_LOC_LIN_REG);

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the cooling rate for mutation
     *
     * @return Cooling rate for mutation as double
     */
    double getScale(){
        return this->scale;
    }
    /**
     * Sets the cooling rate for mutation
     *
     * @param scale as double
     */
    void setScale(double scale) {
        this->scale = scale;
    }

    /**
     * Returns the X coordinates of plane origins as Eigen::VectorXd&
     *
     * @return X coordinate of plane origins as Eigen::VectorXd&
     */
    Eigen::VectorXd& getXO() {
        return this->xO;
    }
    /**
     * @brief setXO
     * @param xO
     */
    void setXO(Eigen::VectorXd& xO) {
        this->xO = xO;
    }

    Eigen::VectorXd& getYO() {
        return this->yO;
    }

    void setYO(Eigen::VectorXd& yO) {
        this->yO = yO;
    }

    Eigen::VectorXd& getZO() {
        return this->zO;
    }

    void setZO(Eigen::VectorXd& zO) {
        this->zO = zO;
    }

    Eigen::VectorXd& getdU() {
        return this->dU;
    }

    void setDU(Eigen::VectorXd& dU) {
        this->dU = dU;
    }

    Eigen::VectorXd& getDL() {
        return this->dL;
    }

    void setDL(Eigen::VectorXd& dL) {
        this->dL = dL;
    }

    double getTheta() {
        return this->theta;
    }

    void setTheta(double theta) {
        this->theta = theta;
    }

    unsigned long getLearningPeriod() {
        return this->learnPeriod;
    }

    void setLearningPeriod(unsigned long lp) {
        this->learnPeriod = lp;
    }

    Eigen::VectorXd& getSurrogateError() {
        return this->surrErr;
    }

    void setSurrogateError(Eigen::VectorXd& se) {
        this->surrErr = se;
    }

    double getSurrogateThreshold() {
        return this->surrThresh;
    }

    void setSurrogateThreshold(double st) {
        this->surrThresh = st;
    }

    unsigned long getMaxLearnNo() {
        return this->maxLearnNo;
    }

    void setMaxLearnNo(unsigned long mln) {
        this->maxLearnNo = mln;
    }

    unsigned long getMinLearnNo() {
        return this->minLearnNo;
    }

    void setMinLearnNo(unsigned long mln) {
        this->minLearnNo = mln;
    }

    Eigen::VectorXd& getCosts() {
        return this->costs;
    }

    void setCosts(Eigen::VectorXd& costs) {
        this->costs = costs;
    }

    Eigen::VectorXd& getROVCurr() {
        return this->rovCurr;
    }

    void setROVCurr(Eigen::VectorXd& rov) {
        this->rovCurr = rov;
    }

    Eigen::MatrixXd& getIARSCurr() {
        return this->iarsCurr;
    }

    void setIARSCurr(Eigen::MatrixXd& iarsCurr) {
        this->iarsCurr = iarsCurr;
    }

    Eigen::MatrixXd& getPopsCurr() {
        return this->popsCurr;
    }

    void setPopsCurr(Eigen::MatrixXd& pc) {
        this->popsCurr = pc;
    }

    Eigen::VectorXd& getUseCurr() {
        return this->useCurr;
    }

    void setUseCurr(Eigen::VectorXd& uc) {
        this->useCurr = uc;
    }

    Eigen::VectorXd& getBest() {
        return this->best;
    }

    void setBest(Eigen::VectorXd& best) {
        this->best = best;
    }

    Eigen::VectorXd& getAverage() {
        return this->av;
    }

    void setAverage(Eigen::VectorXd& av) {
        this->av = av;
    }

    Eigen::MatrixXd& getSurrFit() {
        return this->surrFit;
    }

    void setSurrFit(Eigen::MatrixXd& sf) {
        this->surrFit = sf;
    }

    Eigen::MatrixXd& getIARS() {
        return this->iars;
    }

    void setIARS(Eigen::MatrixXd& iars) {
        this->iars = iars;
    }

    Eigen::MatrixXd& getPops() {
        return this->pops;
    }

    void setPops(Eigen::MatrixXd& pops) {
        this->pops = pops;
    }

    Eigen::MatrixXd& getPopsSD() {
        return this->popsSD;
    }

    void setPopsSD(Eigen::MatrixXd& psd) {
        this->popsSD = psd;
    }

    Eigen::VectorXd& getUse() {
        return this->use;
    }

    void setUse(Eigen::VectorXd& use) {
        this->use = use;
    }

    Eigen::VectorXd& getUseSD() {
        return this->useSD;
    }

    void setUseSD(Eigen::VectorXd& useSD) {
        this->useSD = useSD;
    }

    Eigen::VectorXd& getValues() {
        return this->values;
    }

    void setValues(Eigen::VectorXd& val) {
        this->values = val;
    }

    Eigen::VectorXd& getValuesSD() {
        return this->valuesSD;
    }

    void setValuesSD(Eigen::VectorXd& valSD) {
        this->valuesSD = valSD;
    }

    Eigen::VectorXd getSurrDataValues() {
        return this->values;
    }

    void setSurrDataValues(Eigen::VectorXd& sd) {
        this->values = sd;
    }

    Eigen::VectorXd getSurrDataValuesSD() {
        return this->valuesSD;
    }

    void setSurrDataValuesSD(Eigen::VectorXd& sd) {
        this->valuesSD = sd;    }

    unsigned long getNoSamples() {
        return this->noSamples;
    }

    void setNoSamples(unsigned long ns) {
        this->noSamples = ns;
    }

    RoadGA::Selection getSelector() {
        return this->selector;
    }

    void setSelector(RoadGA::Selection selector) {
        this->selector = selector;
    }

    RoadGA::Scaling getFitScaling() {
        return this->fitScaling;
    }

    void setFitScaling(RoadGA::Scaling fs) {
        this->fitScaling = fs;
    }

    unsigned long getTopProportion() {
        return this->topProp;
    }

    void setTopProportion(unsigned long top) {
        this->topProp = top;
    }

    double getMaxSurvivalRate() {
        return this->maxSurvivalRate;
    }

    void setMaxSurvivalRate(double msr) {
        this->maxSurvivalRate = msr;
    }

    int getTournamentSize() {
        return this->tournamentSize;
    }

    void setTournamentSize(int ts) {
        this->tournamentSize = ts;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /** Virtual Routines *****************************************************/

    /**
     * Extends the 'initialiseExperimentsStorage' routine of Optimiser to
     * include RoadGA specific initialisations.
     */
    virtual void initialiseExperimentStorage();

    /**
     * Extends the 'initialiseStorage' routine of Optimiser to include RoadGA
     * specific initialisations.
     */
    virtual void initialiseStorage();
    /**
     * Creates the initial population used in the optimisation process
     *
     * Creates the initial population for the genetic algorithm that will
     * evolve into a solution.
     *
     * @note This function overrides any base class function of the same name.
     */
    virtual void creation();

    /**
     * Performs crossover for the population used in the optimisation process
     *
     * @param (input) parentsIdx as Eigen::VectorXi&
     * @param (output) children as Eigen::MatrixXd&
     *
     * @note This function overrides any base class function of the same name.
     */
    virtual void crossover(const Eigen::VectorXi &parentsIdx,
            Eigen::MatrixXd& children);

    /**
     * Mutates population used in the optimisation process
     *
     * @param (input) parentsIds as Eigen::VectorXi&
     * @param (output) children as Eigen::MatrixXd&
     *
     * @note This function overrides any base class function of the same name.
     */
    virtual void mutation(const Eigen::VectorXi &parentsIdx,
            Eigen::MatrixXd& children);

    /**
     * Creates the elite children from the current population
     *
     * @param (input) parentsIdx as Eigen::VectorXi&
     * @param (output) children as Eigen::MatrixXd&
     */
    virtual void elite(const Eigen::VectorXi &parentsIdx,
            Eigen::MatrixXd& children);
    /**
     * Runs the optimisation algorithm to devise the best Road
     *
     * If the argument "true" is passed, plots of the best road and
     * optimisation metrics are generated.
     *
     * @param plot as bool (default = false)
     */
    virtual void optimise(bool plot = false);

    /**
     * Evaluates a single generation of the GA
     */
    virtual void evaluateGeneration();

    /**
     * Routine to select between individuals to become parents for the GA
     */
    virtual void selection(Eigen::VectorXi &pc, Eigen::VectorXi &pm,
            Eigen::VectorXi &pe, RoadGA::Selection selector =
            RoadGA::TOURNAMENT);

    /**
     * Returns the outputs at specific stages of the GA
     *
     * This function saves the best road cost, and average cost for each
     * generation.
     */
    virtual void output();

    /**
     * Checks whether the program can stop and if so, under what conditions:
     * 1. 1 = Successfully found solution within stopping criteria
     * 2. 0 = Number of generations exceeded
     * 3. -1 = Fatal error
     *
     * This function also updates the surrogate model if the program needs
     * to run for another generation.
     *
     * @return Stopping flag as int
     */
    virtual int stopCheck();

    /**
     * Creates a default surrogate function for the first iteration of the
     * road design model.
     */
    virtual void defaultSurrogate();

    /**
     * Computes the surrogate function that is updated at each GA iteration
     *
     * @note This function also updates the costs for roads based on the new
     * surrogate as well as records the fitness.
     */
    virtual void computeSurrogate();

    /**
     * Stores the best road to the matrix of bests roads per test
     */
    virtual void assignBestRoad();    

    // Unfortunately, for now the two functions below will result in an extra
    // copy each. We will look to fix this in future iterations.

    /**
     * Computes the data from a full simulation of one road for building the
     * surrogate model for the full traffic flow case.
     *
     * @param (input) road as RoadPtr
     * @param (input) CUDA device number as int (default = 0)
     * @param (output) Matrix of data for building surrogate function for MTE
     * @param (output) Error code as Optimiser::ComputationStatus
     */
    virtual Optimiser::ComputationStatus surrogateResultsMTE(RoadPtr road,
            Eigen::MatrixXd &mteResult, int device = 0);

    /**
     * Computes the data from a full simulation of one road for building the
     * surrogate model for the optimally-controllable traffic flow case.
     *
     * @param (input) road as RoadPtr
     * @param (input) device as int (default = 0)
     * @param (output) Matrix of data for building surrogate function for ROVCR
     * @param (output) Error code as Optimiser::ComputationStatus
     */
    virtual Optimiser::ComputationStatus surrogateResultsROVCR(RoadPtr road,
            Eigen::MatrixXd &rovResult, int device = 0);

    /**
     * Writes out the surrogate results for a road calculation immediately
     * after computing the full model
     *
     * @param data as Eigen::MatrixXd&
     */
    void writeOutSurrogateData(Eigen::MatrixXd& data);

    /**
     * Builds the 2D cubic splines representing the surrogate model for the
     * fixed traffic flow case
     */
    virtual void buildSurrogateModelMTE();

    /**
     * Builds the 2D local linear regressions representing the surrogate model
     * for the fixed traffic flow case
     */
    virtual void buildSurrogateModelMTELocalLinear();

    /**
     * Builds the ND interpolants representing the surrogate model for the
     * controllable traffic flow case
     */
    virtual void buildSurrogateModelROVCR();

    /**
     * Plot routine for visualising the results at each generation
     *
     * @param plot as bool (default = false)
     */
    virtual void plotResults(bool plot = false);

    /**
     * Creates the initial plots for the optimisation before the first
     * generation
     *
     * @param plot as bool (default = false)
     */
    virtual void initialPlots(bool plot = false);

    /**
     * Initialises the program with input data
     *
     * @param inputfile as std::string
     */
    void initialiseFromTextInput(std::string inputFile);

    /**
     * Gets surrogate data for scenario if it already exists
     */
    void getExistingSurrogateData();

    /**
     * Saves the results of the experiment to an external source
     */
    virtual void saveExperimentalResults();

    /**
     * Saves the IARS, Initial Unit Profits, Fixed Costs and Operating Values for all roads
     */
    virtual void saveRunPopulation();

private:
    double scale;               /**< Cooling rate for mutation */
    Eigen::VectorXd xO;         /**< X coordinate of plane origins */
    Eigen::VectorXd yO;         /**< Y coordinate of plane origins */
    Eigen::VectorXd zO;         /**< Z coordinate of plane origins */
    Eigen::VectorXd dU;         /**< Upper limits for plane domains */
    Eigen::VectorXd dL;         /**< Lower limits for plane domains */
    double theta;               /**< Cutting plane angle (to x axis) */
    unsigned long learnPeriod;  /**< Maximum period over which to learn surrogate */
    Eigen::VectorXd surrErr;    /**< Current surrogate standard error */
    double surrThresh;          /**< Maximum surrogate error allowed */
    unsigned long maxLearnNo;   /**< Maximum number of roads on which to perform full model */
    unsigned long minLearnNo;   /**< Minimum number of roads on which to perform full model (best roads found) */
    // Values for current road population /////////////////////////////////////
    Eigen::VectorXd costs;      /**< Total costs of current population */
    Eigen::MatrixXd iarsCurr;   /**< IARs of current population */
    Eigen::MatrixXd popsCurr;   /**< Full traffic flow end pops of current population */
    Eigen::VectorXd useCurr;    /**< Unit operating profit per unit time for all roads */
    Eigen::VectorXd rovCurr;    /**< Utility (ROV) of current population */
    // Values for progression of GA ///////////////////////////////////////////
    Eigen::VectorXd best;       /**< Best cost for each generation */
    Eigen::VectorXd av;         /**< Average cost for each generation */
    Eigen::MatrixXd surrFit;    /**< Standard error of the surrogate model for each generation */
    // Retained values for surrogates models //////////////////////////////////
    Eigen::MatrixXd iars;       /**< IARS for surrogate model (each column for each species) */
    Eigen::MatrixXd pops;       /**< Full traffic flow end pops for surrogate model */
    Eigen::MatrixXd popsSD;     /**< Standard deviations of above */
    Eigen::VectorXd use;        /**< Utilities (ROV) for surrogate models (profit per unit traffic per unit exogenous price factor: petrol, etc.) */
    Eigen::VectorXd useSD;      /**< Utilities standard deviations (to delete) */
    Eigen::VectorXd values;     /**< Mean operating value */
    Eigen::VectorXd valuesSD;   /**< Operating value standard deviation */
    unsigned long noSamples;    /**< Current number of samples available for building surrogates */
    RoadGA::Selection selector; /**< Parents selection routine */
    RoadGA::Scaling fitScaling; /**< Scaling method used for fitness scaling */
    double topProp;             /**< Proportion to consider as top individuals */
    double maxSurvivalRate;     /**< Maximum survival rate for shiftLinearScaling */
    int tournamentSize;         /**< Number of competitors per tournament */

// PRIVATE ROUTINES ///////////////////////////////////////////////////////////

    /**
     * Computes X and Y coordinates of design points for the GA that lie on the
     * perpendicular planes between the start and end points.
     *
     * @param individuals as const long&
     * @param intersectPts as const long&
     * @param startRow as const long&
     */
    void randomXYOnPlanes(const long& individuals, const long &intersectPts,
            const long &startRow);

    /**
     * Computes the X and Y coordinates of design points for the GA, allowing
     * them to lie randomly within the design region.
     *
     * @param individuals as const long&
     * @param intersectPts as const long&
     * @param startRow as const long&
     */
    void randomXYinRegion(const long& individuals, const long& intersectPts,
            const long& startRow);

    /**
     * Computes the Z coordinates of the design points as random elevations
     * within a prescribed permissible range.
     *
     * @param individuals as const long&
     * @param intersectPts as const long&
     * @param startRow as const long&
     * @param population as Eigen::MatrixXd&
     */
    void randomZWithinRange(const long &individuals, const long &intersectPts,
            const long &startRow, Eigen::MatrixXd &population);

    /**
     * Computes the Z coordinates of the design points as close as possible to
     * the actual terrain elevation but within the design grade requirements.
     *
     * @param individuals as const long&
     * @param intersectPts as const long&
     * @param startRow as const long&
     * @param population as Eigen::MatrixXd&
     */
    void zOnTerrain(const long &individuals, const long &intersectPts,
            const long &startRow, Eigen::MatrixXd &population);

    /**
     * Replaces roads with Inf or NaN elements with copies of valid ones.
     *
     * @param roads as Eigen::MatrixXd&
     * @param costs as Eigen::VectorXd&
     */
    void replaceInvalidRoads(Eigen::MatrixXd& roads, Eigen::VectorXd& costs);

    /**
     * Straightens the path between (jj and kk) and (kk and ll)
     *
     * @param ii as int
     * @param jj as int
     * @param kk as int
     * @param ll as int
     * @param children as Eigen::MatrixXd&
     */
    void curveEliminationProcedure(int ii, int jj, int kk, int ll,
            Eigen::MatrixXd& children);

    /**
     * Determines if the function has stalled
     *
     * @return Test result as boolean
     */
    bool stallTest();

    /**
     * Orders and scales the parent roads
     *
     * @param scaleType as RoadGA::Scaling
     * @param (output) parents as Eigen::VectorXi&
     * @param (output) scaling as Eigen::VectorXd&
     */
    void scaling(RoadGA::Scaling scaleType, Eigen::VectorXi& parents,
            Eigen::VectorXd& scaling);

    /**
     * Enables sharing a shared pointer to the derived object from within it
     *
     * @return Shared pointer std::shared_ptr<RoadGA>
     */
    std::shared_ptr<RoadGA> meDerived() {
        return std::static_pointer_cast<RoadGA>(shared_from_this());
    }

    /**
     * Generates n samples of efficently-spaced samples for computing the full
     * models to build the surrogate model.
     *
     * Current and candidate matrices are arranged as M samples x N dimensions.
     *
     * Returns a list of indices corresponding to the selected samples.
     *
     * @param (in) current Eigen::MatrixXd&
     * @param (in) candidates Eigen::MatrixXd&
     * @param (in) n as int
     * @param (out) sample as Eigen::VectorXi&
     */
    void generateSample(const Eigen::MatrixXd& current, const Eigen::MatrixXd&
            candidates, int n, Eigen::VectorXi& sample);
};

#endif
