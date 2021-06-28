#ifndef EXPERIMENTALSCENARIO_H
#define EXPERIMENTALSCENARIO_H

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class ExperimentalScenario;
typedef std::shared_ptr<ExperimentalScenario> ExperimentalScenarioPtr;

class ExperimentalScenario :
        public std::enable_shared_from_this<ExperimentalScenario> {

public:
    // CONSTRUCTORS AND DESTRUCTORS

    /**
     * Constructor I
     *
     * Constructs an %ExperimentalScenario object with default values
     */
    ExperimentalScenario(OptimiserPtr optimiser);
    /**
     * Constructor II
     *
     * Constructs an %ExperimentalScenario object with assigned values
     */
    ExperimentalScenario(OptimiserPtr optimiser, int program, int popLevel,
            int habPrefSD, int lambdaSD, int rangingCoeffSD, int animalBridge,
            int popGR, int popGRSD, int commodity, int commoditySD, int ore,
            int cr, int run);
    /**
     * Destructor
     */
    ~ExperimentalScenario();

    // ACCESSORS///////////////////////////////////////////////////////////////

    /**
     * Returns the Optimiser
     *
     * @return Optimiser as OptimiserPtr
     */
    OptimiserPtr getOptimiser() {
        return this->optimiser.lock();
    }
    /**
     * Sets the Optimiser
     *
     * @param optimiser as OptimiserPtr
     */
    void setOptimiser(OptimiserPtr optimiser) {
        this->optimiser.reset();
        this->optimiser = optimiser;
    }

    /**
     * Returns the program index
     *
     * @return Program index as int
     */
    int getProgram() {
        return this->program;
    }
    /**
     * Sets the program index
     *
     * @param program as int
     */
    void setProgram(int program) {
        this->program = program;
    }

    /**
     * Returns the population level index
     *
     * @return Population level index as int
     */
    int getPopLevel() {
        return this->popLevel;
    }
    /**
     * Sets the population level index
     *
     * @param popLevel as int
     */
    void setPopLevel(int popLevel) {
        this->popLevel = popLevel;
    }

    /**
     * Returns the habitat preference index
     *
     * @return HabPref index as int
     */
    int getHabPref() {
        return this->habPref;
    }
    /**
     * Sets the habitat preference index
     *
     * @param habPref as int
     */
    void setHabPref(int habPref) {
        this->habPref = habPref;
    }

    /**
     * Returns the lambda index as int
     *
     * @return Lamnda index as int
     */
    int getLambda() {
        return this->lambda;
    }
    /**
     * Sets the lambda index as int
     *
     * @param lambda as int
     */
    void setLambda(int lambda) {
        this->lambda = lambda;
    }

    /**
     * Returns the ranging coefficient index as int
     *
     * @return Ranging coefficient index as int
     */
    int getRangingCoeff() {
        return this->rangingCoeff;
    }
    /**
     * Sets the ranging coefficient index as int
     *
     * @param ranging as int
     */
    void setRangingCoeff(int ranging) {
        this->rangingCoeff = ranging;
    }

    /**
     * Returns the animal bridge index as int
     *
     * @return Animal bridge index as int
     */
    int getAnimalBridge() {
        return this->animalBridge;
    }
    /**
     * Sets the animal bridge index as int
     *
     * @param bridge as int
     */
    void setAnimalBridge(int bridge) {
        this->animalBridge = bridge;
    }

    /**
     * Returns the population growth rate mean multiplier index
     *
     * @return Population growth rate mean multiplier index as int
     */
    int getPopGR() {
        return this->popGR;
    }
    /**
     * Sets the population growth rate mean multiplier index
     *
     * @param popGR as int
     */
    void setPopGR(int popGR) {
        this->popGR = popGR;
    }

    /**
     * Returns the population growth rate SD multiplier index
     *
     * @return Population growth rate SD multiplier index as int
     */
    int getPopGRSD() {
        return this->popGRSD;
    }
    /**
     * Sets the population growth rate SD multiplier index
     *
     * @param popGRSD as int
     */
    void setPopGRSD(int popGRSD) {
        this->popGRSD = popGRSD;
    }

    /**
     * Returns the commodity price mean multiplier index as int
     *
     * @return Commodity price mean multiplier index as int
     */
    int getCommodity() {
        return this->commodity;
    }
    /**
     * Sets the commodity price mean multiplier index as int
     *
     * @param commodity as int
     */
    void setCommodity(int commodity) {
        this->commodity = commodity;
    }

    /**
     * Returns the commodity price SD multiplier index as int
     *
     * @return Commodity price SD multiplier index as int
     */
    int getCommoditySD() {
        return this->commodity;
    }
    /**
     * Sets the commodity price SD multiplier index as int
     *
     * @param commodity as int
     */
    void setCommoditySD(int commoditySD) {
        this->commoditySD = commoditySD;
    }

    /**
     * Returns the run number for the current test scenario
     *
     * @return Run as int
     */
    int getRun() {
        return this->run;
    }
    /**
     * Sets the run number of the current test scenario
     *
     * @param run as int
     */
    void setRun(int run) {
        this->run = run;
    }

    /**
     * Returns the index of the ore comosition uncertainty multiplier
     *
     * @return Ore composition standard deviation multiplier as int
     */
    int getOreCompositionSD() {
        return this->oreCompSD;
    }
    /**
     * Sets the index of the ore composition uncertainty multiplier
     *
     * @param ore as int
     */
    void setOreCompositionSD(int ore) {
        this->oreCompSD = ore;
    }

    /**
     * Returns the index (row number) of the current scenario
     *
     * @return Index of current scenario as int
     */
    int getCurrentScenario() {
        return this->currentScenario;
    }
    /**
     * Sets the index (row number) of the current scenario
     *
     * @param cs as int
     */
    void setCurrentScenario(int cs) {
        this->currentScenario = cs;
    }

    /**
     * Returns the index of the comparison road variable cost multiplier used
     *
     * @return Index as int
     */
    int getCompRoad() {
        return this->compRoad;
    }
    /**
     * Sets the index of the comparison road variable cost multiplier used
     *
     * @param cr as int
     */
    void setCompRoad(int cr) {
        this->compRoad = cr;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Computes the row number for the experimental scenario
     */
    void computeScenarioNumber();

    // OPERATORS //////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
private: // (0 = no uncertainty in each of the below)
    std::weak_ptr<Optimiser> optimiser; /**< Calling Optimiser */
    int run;            /**< Current run of scenario */
    int program;        /**< Index of Program used */
    int popLevel;       /**< Index of population level used */
    int habPref;        /**< Index of habitat preference used */
    int lambda;         /**< Index of lambda used */
    int rangingCoeff;   /**< Index of ranging coefficient used */
    int animalBridge;   /**< Index of animal bridge test used */
    int popGR;          /**< Index of population growth uncertainty used */
    int popGRSD;        /**< Index of population growth rate SD used (0 = no uncertainty) */
    int commodity;      /**< Index of commodity price mean multiplier used */
    int commoditySD;    /**< Index of commodity price uncertainty multiplier used */
    int oreCompSD;      /**< Index of ore composition standard deviation multiplier used */
    int compRoad;       /**< Index of comparison road variable cost multiplier used */
    int currentScenario;/**< Index of current scenario. Used for saving results */
};

#endif
