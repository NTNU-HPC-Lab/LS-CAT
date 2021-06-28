#ifndef VERTICALALIGNMENT_H
#define VERTICALALIGNMENT_H

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class VerticalAlignment;
typedef std::shared_ptr<VerticalAlignment> VerticalAlignmentPtr;

/**
 * Class for managing the vertical alignment of a road design
 */
class VerticalAlignment : public std::enable_shared_from_this<VerticalAlignment> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a vertical alignment object with default values
     */
    VerticalAlignment();

    /**
     * Constructor II
     *
     * Constructs a vertical alignment object for a given road and assigns
     * appropriate space.
     */
    VerticalAlignment(RoadPtr road);

    /**
     * Destructor
     */
    ~VerticalAlignment();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns a shared pointer to the road
     *
     * @return Road as RoadPtr
     */
    RoadPtr getRoad() {
        return this->road.lock();
    }

    /**
     * Sets the road
     *
     * @param road as RoadPtr
     */
    void setRoad(RoadPtr road) {
        this->road.reset();
        this->road = road;
    }

    /**
     * Return distances of intersection points along HA
     *
     * @return Distances as const Eigen::VectorXd&
        */
    const Eigen::VectorXd& getSDistances() {
        return this->s;
    }
    /**
     * Sets distances of intersection points along HA
     *
     * @param s as const Eigen::VectorXd&
        */
    void setSDistances(const Eigen::VectorXd& s) {
        this->s = s;
    }
	
    /**
     * Return points of vertical curvature
     *
     * @return PVCs as const Eigen::VectorXd&
        */
    const Eigen::VectorXd& getPVCs() {
        return this->pvc;
    }
    /**
     * Sets points of vertical curvature
     *
     * @param pvcs as const Eigen::VectorXd&
     */
    void setPVCs(const Eigen::VectorXd& pvcs) {
        this->pvc = pvcs;
    }

    /**
     * Return points of vertical tangency
     *
     * @return PVTs as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getPVTs() {
        return this->pvt;
    }
    /**
     * Sets points of vertical tangency
     *
     * @param pvts as const Eigen::VectorXd&
     */
    void setPVTs(const Eigen::VectorXd& pvts) {
        this->pvt = pvts;
    }

    /**
     * Return elevations of points of vertical curvature
     *
     * @return EPVCs as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getEPVCs() {
        return this->epvc;
    }
    /**
     * Sets elevations of points of vertical curvature
     *
     * @param epvcs as const Eigen::VectorXd&
     */
    void setEPVCs(const Eigen::VectorXd& epvcs) {
        this->epvc = epvcs;
    }

    /**
     * Return elevations of points of vertical tangency
     *
     * @return EPVTs as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getEPVTs() {
        return this->epvt;
    }
    /**
     * Sets elevations of points of vertical tangency
     *
     * @param epvts as const Eigen::VectorXd&
     */
    void setEPVTs(const Eigen::VectorXd& epvts) {
        this->epvt = epvts;
    }

    /**
     * Returns polynomial coefficiens (const, x, x^2)
     *
     * @return Polynomial coefficients as const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd& getPolyCoeffs() {
        return this->a;
    }
    /**
     * Sets polynomial coefficients (const, x, x^2)
     *
     * @param coeffs as const Eigen::MatrixXd&
     */
    void setPolyCoeffs(const Eigen::MatrixXd& coeffs) {
        this->a = coeffs;
    }

    /**
     * Return velocities at each IP
     *
     * @return Velocities as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getVelocities() {
        return this->v;
    }
    /**
     * Sets velocities at each IP
     *
     * @param v as const Eigen::VectorXd&
     */
    void setVelocities(const Eigen::VectorXd& v) {
        this->v = v;
    }

    /**
     * Return lengths of curvature
     *
     * @return Lengths of curvature as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getLengths() {
        return this->Ls;
    }
    /**
     * Sets the lengths of curvature
     *
     * @param ls as const Eigen::VectorXd&
     */
    void setLengths(const Eigen::VectorXd& ls) {
        this->Ls = ls;
    }

    /**
     * Return segment grades
     *
     * @return Grades as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getGrades() {
        return this->gr;
    }
    /**
     * Sets segment grades
     *
     * @param gr as const Eigen::VectorXd&
     */
    void setGrades(const Eigen::VectorXd& gr) {
        this->gr = gr;
    }

    /**
     * Return stopping sight distances
     *
     * @return Stopping sight distances as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getSSDs() {
        return this->ssd;
    }
    /**
     * Sets stopping sight distances
     *
     * @param ssds as const Eigen::VectorXd&
     */
    void setSSDs(const Eigen::VectorXd& ssds) {
        this->ssd = ssds;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Computes the vertical alignment of a Road
     */
    void computeAlignment();

private:
    std::weak_ptr<Road> road;   /**< Road */
    Eigen::VectorXd s;          /**< Distance of intersection points along HA */
    Eigen::VectorXd pvc;	/**< S points of vertical curvature */
    Eigen::VectorXd pvt;	/**< S points of vertical tangency */
    Eigen::VectorXd epvc;	/**< Elevations of PVCs */
    Eigen::VectorXd epvt;	/**< Elevations of PVTs */
    Eigen::MatrixXd a;		/**< Polynomial coefficients */
    Eigen::VectorXd v;		/**< Velocities */
    Eigen::VectorXd Ls;		/**< Curvature lengths */
    Eigen::VectorXd gr;		/**< Segment grades */
    Eigen::VectorXd ssd;	/**< Stopping sight distances */

    // PRIVATE ROUTINES ///////////////////////////////////////////////////////
};

#endif
