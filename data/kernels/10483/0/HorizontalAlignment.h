#ifndef HORIZONTALALIGNMENT_H
#define HORIZONTALALIGNMENT_H

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class HorizontalAlignment;
typedef std::shared_ptr<HorizontalAlignment> HorizontalAlignmentPtr;

/**
 * Class for managing horizontal alignment of a road design
 */
class HorizontalAlignment : public std::enable_shared_from_this<HorizontalAlignment> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a horizontal alignment object with default values
     */
    HorizontalAlignment();

    /**
     * Constructor II
     *
     * Constructs a horizontal alignment object for a given road and assigns
     * appropriate space.
     */
    HorizontalAlignment(RoadPtr road);

    /**
     * Destructor
     */
    ~HorizontalAlignment();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the road with this horizontal alignment
     *
     * @return Road as RoadPtr
     */
    RoadPtr getRoad() {
    return this->road.lock();
    }
    /**
     * Sets the road with this horizontal alignment
     *
     * @param road as RoadPtr
     */
    void setRoad(RoadPtr road) {
    this->road.reset();
            this->road = road;
    }

    /**
     * Returns the arc angles (radians)
     *
     * @return Arc angles as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getDeltas() {
        return this->deltas;
    }
    /**
     * Sets the arc angles (radians)
     *
     * @param del as const Eigen::VectorXd&
     */
    void setDeltas(const Eigen::VectorXd& del) {
        this->deltas = del;
    }

    /**
     * Returns the radiis of curvature (radians)
     *
     * @return Radiis of curvature as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getRadii() {
        return this->radii;
    }
    /**
     * Sets the radiis of curvature (radians)
     *
     * @param radii as const Eigen::VectorXd&
     */
    void setRadii(const Eigen::VectorXd radii) {
        this->radii = radii;
    }

    /**
     * Return the required radii of curvature (radians)
     *
     * @return as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getRadiiReq() {
        return this->radiiReq;
    }
    /**
     * Sets the required radii of curvature (radians)
     *
     * @param radiireq as const Eigen::VectorXd&
     */
    void setRadiiReq(const Eigen::VectorXd& radiireq) {
        this->radiiReq = radiireq;
    }

    /**
     * Return the x coordinates of points of curvature (m)
     *
     * @return X coordinates as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getPOCX() {
        return this->pocx;
    }
    /**
     * Sets the x coordinates of points of curvature (m)
     *
     * @param pocx as const Eigen::VectorXd&
     */
    void setPOCX(const Eigen::VectorXd& pocx) {
        this->pocx = pocx;
    }

    /**
     * Returns the y coordinates of points of curvature (m)
     *
     * @return Y coordinates as Eigen::VectorXd&
     */
    const Eigen::VectorXd& getPOCY() {
        return this->pocy;
    }
    /**
     * Sets the y coordinates of points of curvature (m)
     *
     * @param POCY as const Eigen::VectorXd&
     */
    void setPOCY(const Eigen::VectorXd& pocy) {
        this->pocy = pocy;
    }

    /**
     * Returns the x coordinates of the points of tangency (m)
     *
     * @return X coordinates as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getPOTX() {
        return this->potx;
    }
    /**
     * Sets the x coordinates of the points of tangency (m)
     *
     * @param potx as const Eigen::VectorXd&
     */
    void setPOTX(const Eigen::VectorXd& potx) {
        this->potx = potx;
    }

    /**
     * Returns the y coordinates of the points of tangency (m)
     *
     * @return Y coordinates as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getPOTY() {
        return this->poty;
    }
    /**
     * Sets the y coordinates of the points of tangency (m)
     *
     * @param poty as const Eigen::VectorXd&
     */
    void setPOTY(const Eigen::VectorXd& poty) {
        this->poty = poty;
    }

    /**
     * Returns the x coordinates of the chord midpoints (m)
     *
     * @return X midpoints as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getMidX() {
        return this->mx;
    }
    /**
     * Sets the x coordinates of the chord midpoints (m)
     *
     * @param mx as const Eigen::VectorXd&
     */
    void setMidX(const Eigen::VectorXd& mx) {
        this->mx = mx;
    }

    /**
     * Returns the y coordinates of the chord midpoints
     *
     * @return Y midpoints as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getMidY() {
        return this->my;
    }
    /**
     * Sets the y coordinates of the chord midpoints
     *
     * @param my as const Eigen::VectorXd&
     */
    void setMidY(const Eigen::VectorXd& my) {
        this->my = my;
    }

    /**
     * Returns the x coordinates of the centres of curvature (m)
     *
     * @return Centres Xs as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getDelX() {
        return this->delx;
    }
    /**
     * Sets the x coordinates of the centres of curvature (m)
     *
     * @param delx as const Eigen::VectorXd&
     */
    void setDelX(const Eigen::VectorXd& delx) {
        this->delx = delx;
    }

    /**
     * Return the y coordinates of the centres of curvature (m)
     *
     * @return Centre Ys as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getDelY() {
        return this->dely;
    }
    /**
     * Sets the y coordinates of the centres of curvature (m)
     *
     * @param dely as const Eigen::VectorXd&
     */
    void setDelY(const Eigen::VectorXd& dely) {
        this->dely = dely;
    }

    /**
     * Return the design velocities
     *
     * @return Design velocities as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getVelocities() {
        return this->vel;
    }
    /**
     * Sets the design velocities
     *
     * @param vel as const Eigen::VectorXd&
     */
    void setVelocities(const Eigen::VectorXd& vel) {
        this->vel = vel;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Computes the horizontal alignment of a Road
     */
    void computeAlignment();

private:
    std::weak_ptr<Road> road;			/**< Road */
    Eigen::VectorXd deltas;                     /**< Arc angle of curve (rad) */
    Eigen::VectorXd radii;                      /**< Radii of curvature of PIs (rad) */
    Eigen::VectorXd radiiReq;			/**< Required radii of curvature based on desired speed (rad) */
    Eigen::VectorXd pocx;                       /**< Points of curvature, x (m) */
    Eigen::VectorXd pocy;			/**< Points of curvature, y (m) */
    Eigen::VectorXd potx;			/**< Points of tangency, x (m) */
    Eigen::VectorXd poty;			/**< Points of tangency, y (m) */
    Eigen::VectorXd mx;				/**< Chord midpoints, x (m) */
    Eigen::VectorXd my;				/**< Chord midpoints, y (m) */
    Eigen::VectorXd delx;			/**< Centre of curvature, x (m) */
    Eigen::VectorXd dely;			/**< Centre of curvature, y (m) */
    Eigen::VectorXd vel;			/**< Design velocity (m/s) */

    // PRIVATE ROUTINES ///////////////////////////////////////////////////////

    /**
     * Computes the side friction limit based on cornering velocity
     *
     * @param vels as const Eigen::VectorXd&
     * @return fric as Eigen::VectorXd&
     */
    void sideFriction(const Eigen::VectorXd& vels, Eigen::VectorXd&
            fric);

    /**
     * Computes the angle (radians) at the iith intersection point
     *
     * @param xCoords as const Eigen::VectorXd&
     * @param yCoords as const Eigen::VectorXd&
     * @param ii as int
     *
     * @note The first curve has index 1
     */
    double computeDelta(const Eigen::VectorXd& xCoords, const Eigen::VectorXd&
            yCoords, int ii);

    /**
     * Computes the x coordinate of the point of tangency of the iith curve
     *
     * @param rad as const Eigen::VectorXd&
     * @param delta as const Eigen::VectorXd&
     * @param xCoords as const Eigen::VectorXd&
     * @param yCoords as const Eigen::VectorXd&
     * @param ii as int
     * @return POTX as double
     *
     * @note The first curve has index 1
     */
    double computePOTX(const Eigen::VectorXd& rad, const Eigen::VectorXd&
            delta, const Eigen::VectorXd& xCoords, const Eigen::VectorXd&
            yCoords, int ii);

    /**
     * Computes the y coordinate of the point of tangency of the iith curve
     *
     * @param rad as const Eigen::VectorXd&
     * @param delta as const Eigen::VectorXd&
     * @param xCoords as const Eigen::VectorXd&
     * @param yCoords as const Eigen::VectorXd&
     * @param ii as int
     * @return POTY as double
     *
     * @note The first curve has index 1
     */
    double computePOTY(const Eigen::VectorXd& rad, const Eigen::VectorXd&
            delta, const Eigen::VectorXd& xCoords, const Eigen::VectorXd&
            yCoords, int ii);

    /**
     * Computes the x coordinate of the point of curvature of the iith curve
     *
     * @param rad as const Eigen::VectorXd&
     * @param delta as const Eigen::VectorXd&
     * @param xCoords as const Eigen::VectorXd&
     * @param yCoords as const Eigen::VectorXd&
     * @param ii as int
     * @return POCX as double
     *
     * @note The first curve has index 1
     */
    double computePOCX(const Eigen::VectorXd& rad, const Eigen::VectorXd&
            delta, const Eigen::VectorXd& xCoords, const Eigen::VectorXd&
            yCoords, int ii);

    /**
     * Computes the y coordinate of the point of curvature of the iith curve
     *
     * @param rad as const Eigen::VectorXd&
     * @param delta as const Eigen::VectorXd&
     * @param xCoords as const Eigen::VectorXd&
     * @param yCoords as const Eigen::VectorXd&
     * @param ii as int
     * @return POCY as double
     *
     * @note The first curve has index 1
     */
    double computePOCY(const Eigen::VectorXd &rad, const Eigen::VectorXd &delta,
        const Eigen::VectorXd &xCoords, const Eigen::VectorXd &yCoords, int ii);

    /**
     * Computes the absolute distance from the iith POC to the iith PI
     *
     * @param xCoords as const Eigen::VectorXd&
     * @param yCoords as const Eigen::VectorXd&
     * @param pocx as const Eigen::VectorXd&
     * @param pocy as const Eigen::VectorXd&
     * @param ii as int
     * @return Distance as double
     */
    double poc2pi(const Eigen::VectorXd& xCoords, const Eigen::VectorXd&
            yCoords, const Eigen::VectorXd& pocx, const Eigen::VectorXd& pocy,
			int ii);

    /**
     * Computes the absolute distance from the iith POT to the iith PI
     *
     * @param xCoords as const Eigen::VectorXd&
     * @param yCoords as const Eigen::VectorXd&
     * @param pocx as const Eigen::VectorXd&
     * @param pocy as const Eigen::VectorXd&
     * @param ii as int
     * @return Distance as double
     */
    double pot2pi(const Eigen::VectorXd& xCoords, const Eigen::VectorXd&
            yCoords, const Eigen::VectorXd& pocx, const Eigen::VectorXd& pocy,
			int ii);
	
    /**
     * Computes the absolute distance between the iith PI and the
     * previous PI
     *
     * @param xCoords as const Eigen::VectorXd&
     * @param yCoords as const Eigen::VectorXd&
     * @param ii as int
     * @return Distance as double
     */
    double pi2prev(const Eigen::VectorXd& xCoords, const Eigen::VectorXd&
            yCoords, int ii);

    /**
     * Computes the absolute distance between the iith PI and the
     * next PI
     *
     * @param xCoords as const Eigen::VectorXd&
     * @param yCoords as const Eigen::VectorXd&
     * @param ii as int
     * @return Distance as double
     */
    double pi2next(const Eigen::VectorXd &xCoords, const Eigen::VectorXd
        &yCoords, int ii);

    /**
     * Recomputes the radius of curvature at curve ii given a revised POC
     *
     * @param revPOCX as double
     * @param revPOCY as double
     * @param tanlen as double
     * @param xCoords as const Eigen::VectorXd&
     * @param yCoords as const Eigen::VectorXd&
     * @param delta as const Eigen::VectorXd&
     * @param ii as int
     * @return Radius as double
     */
    double radNewPOC(const Eigen::VectorXd& xCoords, const Eigen::VectorXd&
            yCoords, const Eigen::VectorXd& delta, int ii, double tanlen,
            double revPOCX, double revPOCY);

    /**
     * Recomputes the radius of curvature at curve ii given a revised POT
     *
     * @param revPOTX as double
     * @param revPOTY as double
     * @param tanlen as double
     * @param xCoords as const Eigen::VectorXd&
     * @param yCoords as const Eigen::VectorXd&
     * @param delta as const Eigen::VectorXd&
     * @param ii as int
     * @return Radius as double
     */
    double radNewPOT(const Eigen::VectorXd &xCoords, const Eigen::VectorXd
            &yCoords, const Eigen::VectorXd &delta, int ii, double tanlen,
            double revPOTX, double revPOTY);

    /**
     * Computes the x coordinate of the centre of curvature
     *
     * @param xCoords as const Eigen::VectorXd&
     * @param yCoords as const Eigen::VectorXd&
     * @param rad as const Eigen::VectorXd&
     * @param delta as const Eigen::VectorXd&
     * @param mx as const Eigen::VectorXd&
     * @param my as const Eigen::VectorXd&
     * @param ii as int
     * @return X coordinate of curve centre as double
     */
    double curveCentreX(const Eigen::VectorXd& xCoords, const Eigen::VectorXd& yCoords,
            const Eigen::VectorXd& rad, const Eigen::VectorXd& delta, const Eigen::VectorXd& mx,
            const Eigen::VectorXd& my, int ii);

    /**
     * Computes the y coordinate of the centre of curvature
     *
     * @param xCoords as const Eigen::VectorXd&
     * @param yCoords as const Eigen::VectorXd&
     * @param rad as const Eigen::VectorXd&
     * @param delta as const Eigen::VectorXd&
     * @param mx as const Eigen::VectorXd&
     * @param my as const Eigen::VectorXd&
     * @param ii as int
     * @return Y coordinate of curve centre as double
     */
    double curveCentreY(const Eigen::VectorXd &xCoords, const Eigen::VectorXd &yCoords,
            const Eigen::VectorXd &rad, const Eigen::VectorXd &delta, const Eigen::VectorXd &mx,
            const Eigen::VectorXd &my, int ii);
};

#endif
