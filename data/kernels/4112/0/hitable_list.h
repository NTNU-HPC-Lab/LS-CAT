#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

//work with different figures

class hitable_list {
public:
	__device__ hitable_list() {}
	__device__ hitable_list(hitable **l, int n) { 
		list = l; 
		list_size = n; 
	}
	__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
		hit_record tmp_rec;
		bool hit_anything = false;
		float closest = t_max;
		for (int i = 0; i < list_size; i++) {
			if (list[i]->hit(r, t_min, closest, tmp_rec)) {
				hit_anything = true;
				closest = tmp_rec.t;
				rec = tmp_rec;
			}
		}
		return hit_anything;
	}
	hitable **list;
	int list_size;
};

class sphere : public hitable {
public:
	__device__ sphere() {}
	__device__ sphere(float3 cen, float r, bool l) : center(cen), radius(r), isLight(l) {};
	__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
		float3 oc = r.origin() - center;
		float a = dot(r.direction(), r.direction());
		float b = dot(oc, r.direction());
		float c = dot(oc, oc) - radius * radius;
		float d = b * b - a * c;
		if (d > 0) {
			float temp = (-b - sqrt(d)) / a;
			if (temp < t_max && temp > t_min) {
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.isLight = isLight;
				return true;
			}
			temp = (-b + sqrt(d)) / a;
			if (temp < t_max && temp > t_min) {
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.isLight = isLight;
				return true;
			}
		}
		return false;
	}
	float3 center;
	float radius;
	bool isLight;
};

#endif