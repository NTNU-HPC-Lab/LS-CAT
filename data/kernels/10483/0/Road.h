#ifndef ROAD_H
#define ROAD_H

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class Simulator;
typedef std::shared_ptr<Simulator> SimulatorPtr;

class PolicyMap;
typedef std::shared_ptr<PolicyMap> PolicyMapPtr;

class RoadSegments;
typedef std::shared_ptr<RoadSegments> RoadSegmentsPtr;

class RoadCells;
typedef std::shared_ptr<RoadCells> RoadCellsPtr;

class HorizontalAlignment;
typedef std::shared_ptr<HorizontalAlignment> HorizontalAlignmentPtr;

class VerticalAlignment;
typedef std::shared_ptr<VerticalAlignment> VerticalAlignmentPtr;

class Attributes;
typedef std::shared_ptr<Attributes> AttributesPtr;

class Costs;
typedef std::shared_ptr<Costs> CostsPtr;

class Species;
typedef std::shared_ptr<Species> SpeciesPtr;

class SpeciesRoadPatches;
typedef std::shared_ptr<SpeciesRoadPatches> SpeciesRoadPatchesPtr;


/**
 * Class for managing %Road objects
 *
 * This class can also refer to road networks but for convenience and to tie
 * it in with the original PhD code, it is called %Road.
 *
 * In the case of a Network, it is a derived type of %Road. The key differences
 * are that Network contains references to all of the contained roads and it
 * itself does not contain any of the following elements:
 * - RoadSegments
 * - Costs (this is replaced by the costs of contained roads and Program)
 * - HorizontalAlignment
 * - VerticalAlignment
 * - Attributes
 *
 * To perform ROVCR on a network, the different survival probabilities matrices
 * all relate to a different flow configuration through the network. A routine
 * of Network actually finds a suitable set of flow configurations and this is
 * then saved as a type of TrafficProgram (but the flow rates have no meaning)+
 */
class Road : public std::enable_shared_from_this<Road> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs an empty %Road object
     */
    Road();

    /**
     * Constructor II
     *
     * Constructs a %Road object with assigned values
     */
    Road(OptimiserPtr op,
        const Eigen::VectorXd& xCoords, const Eigen::VectorXd& yCoords,
        const Eigen::VectorXd& zCoords);

    /**
     * Constructor III
     *
     * Constructs a %Road object using the encoded genome
     */
    Road(OptimiserPtr op, const Eigen::RowVectorXd& genome);

    /**
     * Destructor
     */
    ~Road();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the Optimiser object
     *
     * @return Optimiser routine as OptimiserPtr
     */
    OptimiserPtr getOptimiser() {
        return this->optimiser.lock();
    }
    /**
     * Sets the Optimiser object
     *
     * @param op as OptimiserPtr
     */
    void setOptimiser(OptimiserPtr op) {
        this->optimiser.reset();
        this->optimiser = op;
    }

    /**
     * Returns the Simulator object
     *
     * @return Simulator as SimulatorPtr
     */
    SimulatorPtr getSimulator() {
        return this->simulator;
    }
    /**
     * Sets the Simulator object
     *
     * @param sim as SimulatorPtr
     */
    void setSimulator(SimulatorPtr sim) {
        this->simulator.reset();
        this->simulator = sim;
    }

    /**
     * Returns the PolicyMap for ROV
     *
     * @return PolicyMap as PolicyMapPtr
     */
    PolicyMapPtr getPolicyMap() {
        return this->policyMap;
    }
    /**
     * Sets the PolicyMap for ROV
     *
     * @param pm as PolicyMapPtr
     */
    void setPolicyMap(PolicyMapPtr pm) {
        this->policyMap.reset();
        this->policyMap = pm;
    }

    /**
     * Returns the road segments from start to end
     *
     * @return Road segments as RoadSegmentsPtr
     */
    RoadSegmentsPtr getRoadSegments() {
        return this->segments;
    }
    /**
     * Sets the road segments from start to end
     *
     * @param segments as RoadSegmentsPtr
     */
    void setRoadSegments(RoadSegmentsPtr segments) {
        this->segments.reset();
        this->segments = segments;
    }

    /**
     * Returns the road cells from start to end
     *
     * @return Road cells as RoadCellsPtr
     */
    RoadCellsPtr getRoadCells() {
        return this->roadCells;
    }
    /**
     * Sets the road cells from start to end
     *
     * @param cells as RoadCellsPtr
     */
    void setRoadCells(RoadCellsPtr cells) {
        this->roadCells.reset();
        this->roadCells = cells;
    }

    /**
     * Returns the HorizontalAlignment
     *
     * @return Horizontal alignment as HorizontalAlignmentPtr
     */
    HorizontalAlignmentPtr getHorizontalAlignment() {
        return this->horizontalAlignment;
    }
    /**
     * Sets the HorizontalAlignment
     *
     * @param ha as HorizontalAlignmentPtr
     */
    void setHorizontalAlignment(HorizontalAlignmentPtr ha) {
        this->horizontalAlignment.reset();
        this->horizontalAlignment = ha;
    }

    /**
     * Returns the VerticalAlignment
     *
     * @return VerticalAlignment as VerticalAlignmentPtr
     */
    VerticalAlignmentPtr getVerticalAlignment() {
        return this->verticalAlignment;
    }
    /**
     * Sets the VerticalAlignment
     *
     * @param va as VerticalAlignmentPtr
     */
    void setVerticalAlignment(VerticalAlignmentPtr va) {
        this->verticalAlignment.reset();
        this->verticalAlignment = va;
    }

    /**
     * Returns the computed road Attributes
     *
     * @return Attributes as AttributesPtr
     */
    AttributesPtr getAttributes() {
        return this->attributes;
    }
    /**
     * Sets the computed road Attributes
     *
     * @param att as AttributesPtr
     */
    void setAttributes(AttributesPtr att) {
        this->attributes.reset();
        this->attributes = att;
    }

    /**
     * Returns the the computed costs
     *
     * @return Computes costs as CostsPtr
     */
    CostsPtr getCosts() {
        return this->costs;
    }
    /**
     * Sets the computed costs
     *
     * @param costs as CostsPtr
     */
    void setCosts(CostsPtr costs) {
        this->costs.reset();
        this->costs = costs;
    }

    /**
     * Returns the habitat patches for each Species for this Road
     *
     * @return SpeciesRoadPatches as cosnt std::vector<SpeciesRoadPatchesPtr>&
     */
    const std::vector<SpeciesRoadPatchesPtr>& getSpeciesRoadPatches() {
        return this->srp;
    }
    /**
     * Sets the habitat patches for each Species for this Road
     *
     * @param srp as const std::vector<SpeciesRoadPatchesPtr>&
     */
    void setSpeciesRoadPatches(const std::vector<SpeciesRoadPatchesPtr>& srp) {
        this->srp = srp;
    }

    /**
     * Returns the test name
     *
     * @return Test name as std::string
     */
    std::string getTestName() {
        return this->testName;
    }
    /**
     * Sets the test name
     *
     * @param tn as std:;string
     */
    void setTestName(std::string tn) {
        this->testName = tn;
    }

    /**
     * Returns X coordinates of the intersection points
     *
     * @return X coordinates as const Eigen::RowVectorXd&
     */
    const Eigen::RowVectorXd& getXCoords() {
        return this->xCoords;
    }
    /**
     * Sets the X coordinates of the intersection points
     *
     * @param xc as const Eigen::RowVectorXd&
     */
    void setXCoords(const Eigen::RowVectorXd& xc) {
        this->xCoords = xc;
    }

    /**
     * Returns the Y coordinates of the intersection points
     *
     * @return Y coordinates as const Eigen::RowVectorXd&
     */
    const Eigen::RowVectorXd& getYCoords() {
        return this->yCoords;
    }
    /**
     * Sets the Y coordinates of the intersection points
     *
     * @param yc as const Eigen::RowVectorXd&
     */
    void setYCoordinates(const Eigen::RowVectorXd& yc) {
        this->yCoords = yc;
    }

    /**
     * Returns the Z coordinates of the intersection points
     *
     * @return Z coordinates of the the intersection points
     */
    const Eigen::RowVectorXd& getZCoords() {
        return this->zCoords;
    }
    /**
     * Sets the Z coordinates of the intersection points
     *
     * @param zc as const Eigen::RowVectorXd&
     */
    void setZCoords(const Eigen::RowVectorXd& zc) {
        this->zCoords = zc;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Design and initialise a road.
     *
     * This routine computes all aspects of a road except operating value
     * 1. computeAlignment
     * 2. plotRoadPath
     * 3. computeRoadCells
     * 4. computeCostElements
     */
    void designRoad();

    /**
     * Evaluate a road.
     *
     * Evaluates a road given a particular lifetime operating Program. If we
     * are learning the surrogate model, this function is passed the value
     * TRUE
     * @param learning as bool (default = false)
     * @param saveResults as bool (default = false)
     * @param device as int (default = 0)
     */
    void evaluateRoad(bool learning = false, bool saveResults = false, int
            device = 0);

    /**
     * Builds the road alignment using the points of intersection. In order,
     * this routine calls:
     * 1. horizontalAlignment
     * 2. verticalAlignment
     */
    void computeAlignment();

    /**
     * Computes the operating costs and animal movement and mortality model
     *
     * Depending on the optimiser options, this call could compute any
     * of the following:
     * 1. Nothing. A simple area-based penalty is applied at an earlier
     *    stage.
     * 2. Full traffic flow for entire horizon.
     * 3. Traffic control.
     *
     * If the function is called in learning mode (learning = true), then the
     * full simulation model for the optimisation scenario is called.
     * Otherwise, the default of using the surrogate function contained in
     * Optimiser->ExperimentalScenario is used.
     *
     * @param learning as bool (default = false)
     * @param saveResults as bool (default = false)
     * @param device as int (default = 0)
     */
    void computeOperating(bool learning = false, bool saveResults = false,
            int device = 0);

    /**
     * Adds simulation patches for a given Species
     *
     * @param srp as SpeciesRoadPatchesPtr
     */
    void addSpeciesPatches(SpeciesPtr species);

    /**
     * Computes the operating profit for fixed traffic flow
     */
    void computeVarProfitICFixedFlow();

private:
    std::weak_ptr<Optimiser> optimiser;         /**< Calling Optimisation object */
    SimulatorPtr simulator;                     /**< Simulator used to produce results */
    PolicyMapPtr policyMap;                     /**< PolicyMap generated from ROV simulation */
    RoadSegmentsPtr segments;                   /**< Road segments */
    RoadCellsPtr roadCells;                     /**< Cells occupied by road */
    HorizontalAlignmentPtr horizontalAlignment; /**< HorizontalAlignment */
    VerticalAlignmentPtr verticalAlignment;     /**< VerticalAlignment */
    AttributesPtr attributes;                   /**< Attributes */
    CostsPtr costs;                             /**< Costs */
    std::vector<SpeciesRoadPatchesPtr> srp;     /**< Patches corresponding to each species */
    std::string testName;                       /**< Name of test */
    Eigen::RowVectorXd xCoords;                 /**< X coordinates of intersection points */
    Eigen::RowVectorXd yCoords;                 /**< Y coordinates of intersection points */
    Eigen::RowVectorXd zCoords;                 /**< Z coordinates of intersection points */
    RoadPtr me();                               /**< Enables sharing from within Road class */

    // PRIVATE ROUTINES ///////////////////////////////////////////////////////

    /**
     * Computes the road cost elements required for valuation: In order, this
     * routine calls:
     * 1. earthworkCost
     * 2. locationCosts
     * 3. lengthCosts
     * 4. accidentCosts
     */
    void computeCostElements();

    /**
     * Computes the road value with the assigned optimisation routine
     */
    //void computeValue();

    /**
     * Computes the grid cells that are occupied by the road
     */
    void computeRoadCells();

    /**
     * Computes the horizontal alignment
     */
    void computeHorizontalAlignment();

    /**
     * Computes the vertical alignment (requires the horizontal alignment to
     * have already been computed).
     */
    void computeVerticalAlignment();

    /**
     * Computes the road path from the horizontal and vertical alignments
     */
    void plotRoadPath();

    /**
     * Creates the simulation patches for each Species in the Region.
     */
    void computeSimulationPatches(bool visualise = false);

    /**
     * Computes the initial animals at risk for a particular Species
     *
     * @param srp as SpeciesRoadPatchesPtr
     * @return iar as Eigen::VectorXd&
     */
    void compueteInitialAAR(SpeciesRoadPatchesPtr srp, Eigen::VectorXd& iar);
};

#endif
