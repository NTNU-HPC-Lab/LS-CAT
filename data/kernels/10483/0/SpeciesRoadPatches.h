#ifndef SPECIESROADPATCHES_H
#define SPECIESROADPATCHES_H

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class Uncertainty;
typedef std::shared_ptr<Uncertainty> UncertaintyPtr;

class Species;
typedef std::shared_ptr<Species> SpeciesPtr;

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class HabitatPatch;
typedef std::shared_ptr<HabitatPatch> HabitatPatchPtr;

/**
 * Class for managing %SpeciesRoadPatches objects
 */
class SpeciesRoadPatches {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a %SpeciesRoadPatches object with default values
     */
    SpeciesRoadPatches(OptimiserPtr optimiser, SpeciesPtr species, RoadPtr road);

    /**
     * Destructor
     */
    ~SpeciesRoadPatches();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the Species
     *
     * @return Species as SpeciesPtr
     */
    SpeciesPtr getSpecies() {
        return this->species;
    }
    /**
     * Sets the Species
     *
     * @param species as SpeciesPtr
     */
    void setSpecies(SpeciesPtr species) {
        this->species.reset();
        this->species = species;
    }

    /**
     * Returns the Road
     *
     * @return Road as RoadPtr
     */
    RoadPtr getRoad() {
        return this->road.lock();
    }
    /**
     * Sets the Road
     *
     * @param road as RoadPtr
     */
    void setRoad(RoadPtr road){
        this->road.reset();
        this->road = road;
    }

    /**
     * Returns the initial Species population in the region
     *
     * @return Initial overall population as double
     */
    double getInitPop() {
        return this->initPop;
    }
    /**
     * Sets the initial Species population in the region
     *
     * @param ip as double
     */
    void setInitPop(double ip) {
        this->initPop = ip;
    }

    /**
     * Returns the habitat patches for simulations
     *
     * @return Habitat patches as const std::vector<HabitatPatchPtr>&
     */
    const std::vector<HabitatPatchPtr>& getHabPatches() {
        return this->habPatch;
    }
    /**
     * Sets the habitat patches during simulations
     *
     * @param habp as const std::vector<HabitatPatchPtr>&
     */
    void setHabPatches(const std::vector<HabitatPatchPtr>& habp) {
        this->habPatch = habp;
    }

    /**
     * Returns the initial populations in each patch
     *
     * @return Initial populations as Eigen::VectorXd&
     */
    const Eigen::VectorXd& getInitPops() {
        return this->initPops;
    }
    /**
     * Sets the initial populations in each patch
     *
     * @param ip as Eigen::VectorXd&
     */
    void setInitPops(Eigen::VectorXd& ip) {
        this->initPops = ip;
    }

    /**
     * Returns the corresponding capacities
     *
     * @return Capacities as Eigen::VectorXd&
     */
    const Eigen::VectorXd& getCapacities() {
        return this->capacities;
    }
    /**
     * Sets the capacities for each patch
     *
     * @param caps as Eigen::VectorXd&
     */
    void setCapacities(Eigen::VectorXd& caps) {
        this->capacities = caps;
    }

    /**
     * Returns the distance from every patch to every other
     *
     * @return Distance matrix as const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd& getDistances() {
        return this->dists;
    }
    /**
     * Sets the distance from every patch to every other
     *
     * @param dist as const Eigen::MatrixXd&
     */
    void setDistances(const Eigen::MatrixXd& dist) {
        this->dists = dist;
    }

    /**
     * Returns the number of crossings between each valid patch transition
     *
     * @return Crossing matrix as const Eigen::MatrixXi&
     */
    const Eigen::MatrixXi getCrossings() {
        return this->crossings;
    }
    /**
     * Sets the number of crossings between each valid patch transition
     *
     * @param cross as const Eigen::MatrixXi&
     */
    void setCrossings(const Eigen::MatrixXi& cross) {
        this->crossings = cross;
    }
    /**
     * Returns the transition probability matrix
     *
     * @return Transition probability matrix as const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd& getTransProbs() {
        return this->transProbs;
    }
    /**
     * Sets the transition probability matrix
     *
     * @param transProbs as const Eigen::MatrixXd&
     */
    void setTransProbs(const Eigen::MatrixXd& transProbs) {
        this->transProbs = transProbs;
    }

    /**
     * Returns the survival probability matrices for each control
     *
     * @return Survival probability matrices as const std::vector<Eigen::MatrixXd>&
     */
    const std::vector<Eigen::MatrixXd>& getSurvivalProbs() {
        return this->survProbs;
    }
    /**
     * Sets the survival probability matrices for each control
     *
     * @param survProbs as const std::vector<Eigen::MatrixXd>&
     */
    void setSurvivalProbs(const std::vector<Eigen::MatrixXd>& survProbs) {
        this->survProbs = survProbs;
    }

    /**
     * Returns the mean end animal population about the road
     *
     * @return Mean end animal population as double
     */
    double getEndPopMean() {
        return this->endPopMean;
    }
    /**
     * Sets the mean end animal population about the road
     *
     * @param pop as double
     */
    void setEndPopMean(double pop) {
        this->endPopMean = pop;
    }

    /**
     * Returns the end animal population standard deviation about the road
     *
     * @return End animal population standard deviation as double
     */
    double getEndPopSD() {
        return this->endPopSD;
    }
    /**
     * Sets the end animal population standard deviation about the road
     *
     * @param pop as double
     */
    void setEndPopSD(double pop) {
        this->endPopSD = pop;
    }

    /**
     * Returns the initial proportion of animals at risk for the road and species
     *
     * @return Initial animalas at risk as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getInitAAR() {
        return this->initAAR;
    }
    /**
     * Sets the initial proportion of animals at risk for the road and species
     *
     * @param aar as double
     */
    void setInitAAR(const Eigen::VectorXd& aar) {
        this->initAAR = aar;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Builds the animal movement and mortality model, initialising patches,
     * crossings, distances, transition and survival probabilities.
     */
    void createSpeciesModel(bool visualise = false);

    /**
     * Computes the AAR and expected population for each control
     *
     * Computes the animals at risk (AAR) percentage and expected population
     * This value is not stochastic. It takes into account all expected
     * transition probabilities and survival probabilities of each transition
     * as well as the prevailing population at the start of a period (i.e.
     * before accounting for natural births and deaths).
     *
     * @param pops as const Eigen::VectorXd&
     * @return AAR vector as Eigen::VectorXd&
     */
    void computeAAR(const Eigen::VectorXd& pops, Eigen::VectorXd& aar);

    /**
     * Computes the AARs for this species and road using initial populations
     */
    void computeInitialAAR();

///////////////////////////////////////////////////////////////////////////////
private:
    SpeciesPtr species;                     /**< Speices used */
    double initPop;                         /**< Initial species population */
    std::weak_ptr<Road> road;               /**< Corresponding road */
    std::vector<HabitatPatchPtr> habPatch;  /**< Corresponding habitat patches */
    Eigen::VectorXd initPops;               /**< Initial patch populations */
    Eigen::VectorXd capacities;             /**< Corresponding patch capacities */
    Eigen::MatrixXd dists;                  /**< Distances between patches */
    Eigen::MatrixXi crossings;              /**< Number of crossings for each journey */
    Eigen::MatrixXd transProbs;             /**< Transition probabilities. Rows sum to 1 */
    std::vector<Eigen::MatrixXd> survProbs; /**< Survival probabilities. Rows multiply to <= 1*/
    double endPopMean;                      /**< Mean end population (for run to extinction) */
    double endPopSD;                        /**< End population standard deviation */
    Eigen::VectorXd initAAR;                /**< Initial AAR of the species for each control */

    // PRIVATE ROUTINES ///////////////////////////////////////////////////////

    /**
     * Generates the habitat patches relating to this species for the road in
     * question using a basic grid method.
     *
     * If we are visualising the results, we pass the argument "true"
     *
     * @param visualise as bool
     */
    void generateHabitatPatchesGrid(bool visualise = false);

    /**
     * Generate habitat patches relating to this species using a more advanced
     * blob method that ought to be more computationally tractable.
     *
     * This method first looks at the covexity and size of each blob in
     * determining if the blob should be split into multiple patches or be left
     * as a single patch.
     */
    void generateHabitatPatchesBlob();

    /**
     * Computes the distance between each patch with every other patch.
     */
    void habitatPatchDistances();

    /**
     * Computes the number of road crossings between each patch.
     */
    void roadCrossings();

    /**
     * Computes the transition probabilities for this road
     */
    void computeTransitionProbabilities();

    /**
     * Computes the survival probabilities for each transition for this road
     */
    void computeSurvivalProbabilities();
};

#endif
