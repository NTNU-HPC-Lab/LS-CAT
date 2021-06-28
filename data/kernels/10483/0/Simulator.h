#ifndef SIMULATOR_H
#define SIMULATOR_H

class MonteCarloROV;
typedef std::shared_ptr<MonteCarloROV> MonteCarloROVPtr;

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class SpeciesRoadPatches;
typedef std::shared_ptr<SpeciesRoadPatches> SpeciesRoadPatchesPtr;

class Simulator;
typedef std::shared_ptr<Simulator> SimulatorPtr;

/**
 * Class for managing simulations
 */
class Simulator : public MonteCarloROV, 
        public std::enable_shared_from_this<Simulator> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Default, blank constructor
     */
    Simulator();

    /**
     * Constructor II
     *
     * Pass the Road as an argument for initialisation
     */
    Simulator(RoadPtr road);

    /**
     * Destructor
     */
    ~Simulator();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the road calling the simulator
     *
     * @return Road as RoadPtr
     */
    RoadPtr getRoad() {
        return this->road.lock();
    }
    /**
     * Sets the road calling the simulator
     *
     * @param road as RoadPtr
     */
    void setRoad(RoadPtr road) {
        this->road.reset();
        this->road = road;
    }

    /**
     * Returns the end population from all runs
     *
     * @return End populations as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getEndPops() {
        return this->endPops;
    }
    /**
     * Sets the end population from all runs
     *
     * @param endPops as const Eigen::VectorXd&
     */
    void setEndPops(const Eigen::VectorXd& endPops) {
        this->endPops = endPops;
	}

    /**
     * Returns the initial animals at risk
     *
     * @return Initial animals at risk as double
     */
    double getIAR() {
        return this->initAAR;
    }
    /**
     * Sets the initial animals at risk
     *
     * @param iar as double
     */
    void setIAR(double iar) {
        this->initAAR = iar;
    }

    /**
     * Returns the extinction dollar cost penalty
     *
     * @return Extionction penalty as double
     */
    double getPenalty() {
        return this->penalty;
    }
    /**
     * Sets the extinction penalty
     *
     * @param ep as double
     */
    void setPenalty(double ep) {
        this->penalty = ep;
    }
    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////
    // These simulation routines are only called when learning the surrogate
    // model used after each generation in the optimisation routine.

    /**
     * Runs the full flow simulation over the entire design horizon for all
     * species encountered by the road.
     *
     * This method computes the expected end populations and their standard
     * deviations that are then used to compute the road value.
     *
     * If the calling routine wishes to keep the population in each patch over
     * time (i.e. so as to visualise the progression of a single population
     * path) we pass in a reference to a matrix. If we do not enter a matrix,
     * then we only produce the end metrics but not a realised simulation path
     * that can be visualised via a post-processing routine.
     *
     * @param device as int (default = 0)
     * @return Computation status as Optimiser::ComputationStatus
     */
    Optimiser::ComputationStatus simulateMTE(int device = 0);

    /**
     * Overloaded version of the preceding function for storing an entire
     * single path for visualisation
     *
     * @param (output) visualiseResults as std::vector<Eigen::MatrixXd>&
     */
    void simulateMTE(std::vector<Eigen::MatrixXd> &visualiseResults);

    /**
     * Runs the optimally controlled traffic scenario over the entire design
     * horizon for all species encountered by the road.
     *
     * We want to override the parent class function
     *
     * @param policyMap as bool (default = false)
     * @param device as int (default = 0)
     */
    virtual void simulateROVCR(bool policyMap = false, int device = 0);

    /**
     * Overloaded version of the preceding function for storing an entire
     * single realisation for visualisation. This keeps the populations in
     * each patch over time as well as the traffic control choice.
     *
     * @param (output) visualisePops as std::vector<Eigen::MatrixXd>&
     * @param (output) visualiseFlows as Eigen::VectorXi&
     * @param (output) visualiseUnitProfits as Eigen::VectorXd&
     * @note This method automatically useds ALGO6, which is the most
     * comprehensive method, to visualise a single Monte Carlo chain and the
     * optimal actions taken over time.
     * @note The policy map for this road has to have already been built
     */
    virtual void simulateROVCR(std::vector<Eigen::MatrixXd>& visualisePops,
            Eigen::VectorXi &visualiseFlows, Eigen::VectorXd&
            visualiseUnitProfits);

private:
    std::weak_ptr<Road> road;   /**< Road owning simulator */
    Eigen::VectorXd endPops;    /**< End populations from all sims */
    double initAAR;             /**< Initial animals at risk */
    double penalty;             /**< Extinction penalty */
    SimulatorPtr me();          /**< Enables sharing from within Simulator class */

    // PRIVATE ROUTINES ///////////////////////////////////////////////////////

    /**
     * This step is performed at every time step of the animal simulation model
     * to account for competition between species. It is called after animal
     * movement and mortality related to roads but before accounting for
     * natural birth and death.
     *
     * @note This function is currently not in use
     */
    void animalCompetition();

    /**
     * This is the last step performed at each time step of the animal simulation
     * model. It accounts for natural birth and death in each habitat patch after
     * the effects of road movement and mortality and species competition have
     * taken place.
     *
     * @param (input) species as const SpeciesRoadPatchesPtr
     * @param (input) capacities as const Eigen::VectorXd&
     * @param (input/output) pops as Eigen::VectorXd&
     */
    void naturalBirthDeath(const SpeciesRoadPatchesPtr species, const
            Eigen::VectorXd& capacities, Eigen::VectorXd& pops);

    /**
     * Simulates a single path up to the design horizon for one species for MTE
     *
     * This overloaded function returns an entire single path for all patches
     *
     * @param (input) species as const std::vector<SpeciesRoadPatchesPtr>&
     * @param (input) initPops as const std::vector<Eigen::VectorXd>&
     * @param (input) capacities as const std::vector<Eigen::VectorXd>&
     * @param (output) finalPops as Eigen::MatrixXd&
     */
    void simulateMTEPath(const std::vector<SpeciesRoadPatchesPtr>& species,
            const std::vector<Eigen::VectorXd>& initPops, const
            std::vector<Eigen::VectorXd>& capacities,
            std::vector<Eigen::VectorXd>& finalPops);

    /**
     * Simulates a single path up to the design horizon for all species
     *
     * This overloaded function returns an entire single path for all patches.
     * This routine is used for visualising a single possible outcome for one
     * species in the region.
     *
     * @param (input) species as const std::vector<SpeciesRoadPatchesPtr>&
     * @param (input) initPops as const std::vector<Eigen::VectorXd>&
     * @param (input) capacities as const std::vector<Eigen::VectorXd>&
     * @param (output) visualiseResults as std::vector<Eigen::MatrixXd>&
     */
    void simulateMTEPath(const std::vector<SpeciesRoadPatchesPtr>& species,
            const std::vector<Eigen::VectorXd>& initPops, const
            std::vector<Eigen::VectorXd> &capacities,
            std::vector<Eigen::MatrixXd> &visualiseResults);

    /**
     * Simulates a single forward path for Real Options, returning the values
     * for each uncertainty at each time for the time steps.
     *
     * This method is used for finding the optimal policy for a road and is
     * called many times as part of a Monte Carlo simulation. It does not save
     * the population in each patch for each species, hence it is not used for
     * visualisation of the optimal policy.
     *
     * @note Time steps refers to the number of time steps up to and including
     * the end time step, T. Therefore, the algorithm starts at time step
     * t = T -  timeSteps. I.e. if we want to compute the full path to the very
     * beginning, t = T. In addition, we must always pass in the full vector,
     * even if we are only changing a portion of the time steps.
     *
     * @param (input) species as const std::vector<SpeciesRoadPatchesPtr>&
     * @param (input) initPops as const std::vector<Eigen::VectorXd>&
     * @param (input) capacities as const std::vector<Eigen::VectorXd>&
     * @param (output) exogenousPaths as std::vector<Eigen::VectorXd>&
     * @param (output) endogenousPaths as std::vector<Eigen::VectorXd>&
     */
    virtual void simulateROVCRPath(const std::vector<SpeciesRoadPatchesPtr>&
            species, const std::vector<Eigen::VectorXd>& initPops, const
            std::vector<Eigen::VectorXd>& capacities,
            std::vector<Eigen::VectorXd>& exogenousPaths,
            std::vector<Eigen::VectorXd>& endogenousPaths);

    /**
     * Simulates a single full forward path for Real Options, returning the
     * values for each uncertainty (including individual patch populations)
     * for the entire time horizon.
     *
     * @note This method is used for visualisation purposes ONLY
     *
     * @param (input) species as const std::vector<SpeciesRoadPatchesPtr>&
     * @param (input) initPops as const std::vector<Eigen::VectorXd>&
     * @param (input) capacities as const std::vector<Eigen::VectorXd>&
     * @param (input) exogenousPaths const as std::vector<Eigen::VectorXd>&
     * @param (output) endogenousPaths as std::vector<Eigen::VectorXd>&
     * @param (output) visualiseResults as std::vector<Eigen::MatrixXd>&
     */
    virtual void simulateROVCRPath(const std::vector<SpeciesRoadPatchesPtr>&
            species, const std::vector<Eigen::VectorXd>& initPops, const
            std::vector<Eigen::VectorXd>& capacities,
            const std::vector<Eigen::VectorXd>& exogenousPaths,
            std::vector<Eigen::VectorXd>& endogenousPaths,
            std::vector<Eigen::MatrixXd> &visualiseResults);

    /**
     * Recomputes a single path from a specific time to the end time, T
     *
     * This function uses the stored optimal profits to go for each control
     * at each time step to recompute the forward paths from a specific time.
     *
     * @note We pass in the full vectors (i.e. from time t = 0 to t = T) and
     * only update the last 'T - timeStep' entries.
     *
     * @param (input) species as const std::vector<SpeciesRoadPatchesPtr>&
     * @param (input) initPops as const std::vector<Eigen::VectorXd>&
     * @param (input) capacities as const std::vector<Eigen::VectorXd>&
     * @param (input) exogenousPaths const as std::vector<Eigen::VectorXd>&
     * @param (input) timeStep as const unsigned long
     * @param (input) optPtG as std::vector<std::vector<alglib::spline1dinterpolant>>
     * @param (output) endogenousPaths as std::vector<Eigen::VectorXd>&
     */
    virtual void recomputeForwardPath(const std::vector<SpeciesRoadPatchesPtr>&
            species, const std::vector<Eigen::VectorXd>& initPops, const
            std::vector<Eigen::VectorXd>& capacities,
            const std::vector<Eigen::VectorXd>& exogenousPaths,
            const unsigned long timeStep, const std::vector<std::vector<
            alglib::spline1dinterpolant>> optPtG,
            std::vector<Eigen::VectorXd>& endogenousPaths);
};

#endif
