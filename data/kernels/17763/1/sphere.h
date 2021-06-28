#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class Sphere: public Hitable {
    public:
        __device__ Sphere(){};
        __device__ Sphere(Vector3 cen, float r):center(cen), radius(r) {};
        __device__ virtual bool hit(const Ray& r, float tmin, float tmax, hit_record& rec) const;
        Vector3 center;
        float radius;
};

__device__ bool Sphere::hit(const Ray& r, float tmin, float tmax, hit_record& rec) const {
    Vector3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < tmax && temp > tmin) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < tmax && temp > tmin) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            return true;
        }
    }
    return false;
}

#endif