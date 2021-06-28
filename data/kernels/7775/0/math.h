#pragma once
#define ANG2RAD 3.14159265358979323846/180.0
#include "Transform.h"

namespace Math
{
	struct Plane
	{
		float a, b, c, d;

		void Normalize();
		float PlaneDot(glm::vec3 point);

		static Plane PlaneFromPoints(glm::vec3 pv1, glm::vec3 pv2, glm::vec3 pv3)
		{
			glm::vec3 edge1, edge2, normal, Nnormal;

			edge1.x = 0.0f; edge1.y = 0.0f; edge1.z = 0.0f;
			edge2.x = 0.0f; edge2.y = 0.0f; edge2.z = 0.0f;

			edge1 = pv2 - pv1;
			edge2 = pv3 - pv1;
			normal = glm::cross(edge1, edge2);
			Nnormal = glm::normalize(normal);
			return PlaneFromPointNormal(pv1, Nnormal);
		};
		static Plane PlaneFromPointNormal(glm::vec3 pvpoint, glm::vec3 pvnormal)
		{
			Plane pout;
			pout.a = pvnormal.x;
			pout.b = pvnormal.y;
			pout.c = pvnormal.z;
			pout.d = -glm::dot(pvpoint, pvnormal);
			return pout;
		}
	};
	struct Sphere;
	struct Box
	{
		glm::vec3 center;
		glm::vec3 halfSize;
		
		bool Contain(Sphere& sphere);
		bool Collision(Box& boundingBox);
		glm::vec3 getVertex(int corner);
	};
	struct Sphere
	{
		glm::vec3 center;
		float radius;
		
		bool Collision(Sphere& sphere);

		static Sphere CreateBoundingSphere(VertexBuffer* vertexBuffer);
	};
};

class FrustumG {

private:

	enum {
		TOP = 0, BOTTOM, LEFT,
		RIGHT, NEARP, FARP
	};

public:

	static enum { OUTSIDE, INTERSECT, INSIDE };
	Math::Plane pl[6];
	glm::vec3 ntl, ntr, nbl, nbr, ftl, ftr, fbl, fbr;
	float nearD, farD, ratio, angle, tang;
	float nw, nh, fw, fh;

	FrustumG() {}
	~FrustumG() {}

	void setCamInternals(float angle, float ratio, float nearD, float farD);
	void setCamDef(Transform& trans);
	int pointInFrustum(glm::vec3 &p);
	int sphereInFrustum(Math::Sphere& sphere);
	int boxInFrustum(Math::Box& b);
};