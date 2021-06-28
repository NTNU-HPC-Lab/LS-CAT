#ifndef MODEL_H
#define MODEL_H
#include <vector>
#include <stack>
#include <string>
#include <map>
#define NEAR 0.1

std::vector<float> sumfv(std::vector<float> v1,
    const std::vector<float> &v2);
std::vector<float> subfv(std::vector<float> v1,
    const std::vector<float> &v2);
std::vector<float> mulfv(std::vector<float> v1,
    const std::vector<float> &v2);
std::vector<float> upscafv(float c, std::vector<float> v);
std::vector<float> doscafv(float c, std::vector<float> v);
std::vector<float> expfv(std::vector<float> v);
float dotfv(const std::vector<float> &v1,
  const std::vector<float> &v2);
std::vector<float> crossf3(const std::vector<float> &u, 
    const std::vector<float> &v);
float magfv(const std::vector<float> &v);
std::vector<float> normfv(const std::vector<float> &v);
float accfv(const std::vector<float> &v);
void printVector(const std::vector<float> &v);
std::vector<float> refract(const std::vector<float> &d,
 const std::vector<float> &n, float rori);


enum DMatType {LAMBERT, PHONG, REFLECTIVE, DIELECTRIC, TEXTURE};
enum DLigType {PUNCTUAL, SPOT, AMBIENT, DIRECTIONAL};
enum DOctType {BRANCH, LEAVE};

struct DMaterial {
  float col[3], att[3], param, *tex;
  DMatType type;
};

struct DSphere {
  float c[3], r;
  int nMats;
  DMaterial **mats;
};

struct DSpheres {
  DSphere **s;
  unsigned short nSpheres;
};

struct DLight {
  float col[3], pos[3], dir[3], angle;
  DLigType type;
};

struct DLights {
  DLight **l;
  unsigned short nLights;
};

struct DCamera {
  float u[3],v[3],w[3],pos[3], 
        left,right,bottom,top,
        scene_refr,bg_col[3],
        near,epsilon,inf;
  unsigned short width,height,ray_depth;
  bool shadows;
};

struct DTriangle {
  float v[3][3], uv[3][2], n[3][3];
  bool uvA, nA; int nMats;
  DMaterial **mats;
};

struct DOctTree {
  DOctTree **child;
  DTriangle **triangles;
  unsigned short nTriangles;
  float ini[3], end[3];
  DOctType type;
};

struct DRay {
  float eye[3], dir[3],t,iPoint[3],n[3],col[3],att[3],level[3];
  DSphere *s; DTriangle *tri; DOctTree *oct;
  unsigned short depth,nLevel;
};

struct DStack {
  DRay ray[10];
  unsigned short size;
};

// child accounts for current sub node for every node
// if it gets to 9, it contains not what is being searched for
struct DOctStack {
  DOctTree *oct[20];
  unsigned short size, child[20];
};



class Material {

  protected:
    std::vector<float> _color;
  
  public:
    virtual ~Material(){};
    virtual void colorate(std::vector<float> &color,
       const std::vector<float> &iSphere,
       const std::vector<float> &n,
       const std::vector<float> &d, std::stack<float> &level,
       unsigned char depth, float t)
      const = 0;
    virtual DMaterial* buildDMaterial() const = 0;
};

class LambertMaterial : public Material {

  public:
    LambertMaterial(std::vector<float> color) { _color = color; }
    virtual ~LambertMaterial(){};
    void colorate(std::vector<float>&color,
        const std::vector<float>&iSphere,
        const std::vector<float>&n,
        const std::vector<float>&d, std::stack<float> &level,
        unsigned char depth, float t)
      const override;
    DMaterial* buildDMaterial() const override;
};

class PhongMaterial : public Material {

  private:
    float _shi;

  public:
    PhongMaterial(std::vector<float> color, float shi):
      _shi(shi) { _color=color; }
    virtual ~PhongMaterial(){};
    void colorate(std::vector<float>&color,
        const std::vector<float>&iSphere,
        const std::vector<float>&n, const std::vector<float>&d,
        std::stack<float> &level, unsigned char depth, float t)
      const override;
    DMaterial* buildDMaterial() const override;
};

class ReflectiveMaterial : public Material {

  public:
    ReflectiveMaterial(std::vector<float> color) { _color=color; }
    virtual ~ReflectiveMaterial(){};
    void colorate(std::vector<float>&color,
        const std::vector<float>&iPoint,
        const std::vector<float> &n,
        const std::vector<float>&d, std::stack<float> &level,
        unsigned char depth, float t) const override;
    DMaterial* buildDMaterial() const override;
};

class DielectricMaterial : public Material {

  private:
    std::vector<float> _att;
    float _refr;

  public:
    DielectricMaterial(std::vector<float> color,
        std::vector<float> att, float refr):
      _att(att), _refr(refr) { _color=color; }
    virtual ~DielectricMaterial(){}
    void colorate(std::vector<float>&color,
        const std::vector<float>&iPoint,
        const std::vector<float> &n,
        const std::vector<float>&d, std::stack<float> &level,
        unsigned char depth, float t) const override;
    DMaterial* buildDMaterial() const override;
};

class TextureMaterial : public Material {

  private:
    float *d_tex; std::string _file;
    unsigned short _W,_H;

  public:
    TextureMaterial(std::string mat_path);
    ~TextureMaterial();
    void colorate(std::vector<float>&color,
        const std::vector<float>&iPoint,
        const std::vector<float> &n,
        const std::vector<float>&d, std::stack<float> &level,
        unsigned char depth, float t) const override;
    DMaterial* buildDMaterial() const override;
};


class Light {

  protected:
    std::vector<float> _color;

  public:
    Light(const std::vector<float> color): _color(color) {}
    virtual ~Light(void){};
    virtual void iluminateLambert(
        std::vector<float> &color,
        const std::vector<float>&mat_color,
        const std::vector<float>&iSphere,
        const std::vector<float> &n) = 0;
    virtual void iluminatePhong(
        std::vector<float> &color,
        const std::vector<float> &mat_color,
        const std::vector<float> &iSphere,
        const std::vector<float> &n,
        const std::vector<float> &d, int shi) = 0;
    virtual DLight* buildDLight(void) const = 0;
};

class PunctualLight : public Light {
    
  private:
    std::vector<float> _pos;

  public:
    PunctualLight(std::vector<float> color,
        std::vector<float> pos): Light(color), _pos(pos) {}
    virtual ~PunctualLight(void){}
    void iluminateLambert(
        std::vector<float> &color,
        const std::vector<float>&mat_color,
        const std::vector<float>&iPoint,
        const std::vector<float> &n)
      override;
    void iluminatePhong(
        std::vector<float> &color,
        const std::vector<float> &mat_color,
        const std::vector<float> &iPoint,
        const std::vector<float> &n,
        const std::vector<float> &d, int shi) override;
    DLight* buildDLight(void) const override; 
};

class SpotLight : public Light {
    
  private:
    std::vector<float> _pos, _dir;
    float _angle;

  public:
    SpotLight(std::vector<float> color, std::vector<float> pos,
        float angle, std::vector<float> dir): Light(color),
      _pos(pos), _dir(dir), _angle(angle) {}
    virtual ~SpotLight(void){}
    void iluminateLambert(
        std::vector<float> &color,
        const std::vector<float>&mat_color,
        const std::vector<float>&iPoint,
        const std::vector<float> &n)
      override;
    void iluminatePhong(
        std::vector<float> &color,
        const std::vector<float> &mat_color,
        const std::vector<float> &iPoint,
        const std::vector<float> &n,
        const std::vector<float> &d, int shi) override;
    DLight* buildDLight(void) const override; 
};

class AmbientLight : public Light {
  
  public:
    AmbientLight(std::vector<float> color):
      Light(color) {}
    virtual ~AmbientLight(void) {}
    void iluminateLambert(
        std::vector<float> &color,
        const std::vector<float>&mat_color,
        const std::vector<float>&iPoint,
        const std::vector<float> &n)
      override { color=sumfv(color,mulfv(_color,mat_color)); }
    void iluminatePhong(
        std::vector<float> &color,
        const std::vector<float> &mat_color,
        const std::vector<float> &iPoint,
        const std::vector<float> &n,
        const std::vector<float> &d, int shi) override {}
    DLight* buildDLight(void) const override; 
};

class DirectionalLight : public Light {
    
  private:
    std::vector<float> _dir;

  public:
    DirectionalLight(std::vector<float> color,
        std::vector<float> dir): Light(color),
      _dir(normfv(dir)) {}
    virtual ~DirectionalLight(void){}
    void iluminateLambert(
        std::vector<float> &color,
        const std::vector<float>&mat_color,
        const std::vector<float>&iPoint,
        const std::vector<float> &n)
      override;
    void iluminatePhong(
        std::vector<float> &color,
        const std::vector<float> &mat_color,
        const std::vector<float> &iPoint,
        const std::vector<float> &n,
        const std::vector<float> &d, int shi) override;
    DLight* buildDLight(void) const override; 
};



class Object {

  protected:

    std::vector<const Material *> _mats;

  public:

    Object(void){};
    virtual ~Object(void){};
    virtual bool intersectsRay(const std::vector<float> &e,
        const std::vector<float> &d, std::vector<float> &iSphere,
        std::vector<float> &n, float &t,
        std::vector<const Material*>&mats) = 0;
    virtual bool intersectsOctNode(const std::vector<float>&ini,
        const std::vector<float>&end)=0;
    const std::vector<const Material*>& getMaterials(void) const {
      return _mats;
    }
};
class Vertex; class Mesh;

class Triangle: public Object {

  public:
    std::vector<Vertex*> _v;
    std::vector<float> _n; Mesh *_mesh;
    Triangle(std::vector<Vertex*> &v);
    bool intersectsRay(const std::vector<float>&e,
        const std::vector<float>&d, std::vector<float>&iPoint,
        std::vector<float>&n, float&t,
        std::vector<const Material*>&mats) override;
    bool intersectsOctNode(const std::vector<float>&ini,
        const std::vector<float>&end);
    void setN();
};

class Vertex {

  private:
    std::vector<const Triangle *> _t;

  public:
    std::vector<float> _p,_uv,_n;
    Vertex(const std::vector<float> &p): _p(p) {}
    void addT(const Triangle *t){ _t.push_back(t); }
    void setN(void){
      std::vector<float> n(3,0);
      for (auto &t: _t) n = sumfv(n,t->_n);
      _n = normfv(n);
    }
};


class OctNode: public Object {

  protected:
    std::vector<float> _ini, _end;

  public:
    virtual void insert(Object *obj)=0;
    virtual bool intersectsRay(const std::vector<float> &e,
      const std::vector<float> &d, std::vector<float> &iP,
      std::vector<float> &n, float &t,
      std::vector<const Material*>&mats);
    virtual bool intersectsChildren(const std::vector<float> &e,
      const std::vector<float> &d, std::vector<float> &iP,
      std::vector<float> &n, float &t,
      std::vector<const Material*>&mats)=0;

    bool intersectsObj(Object *obj){
      return obj->intersectsOctNode(_ini, _end);}
    bool intersectsOctNode(const std::vector<float>&ini,
        const std::vector<float>&end){return false;}
    virtual void countNode(int &count) const = 0;
    virtual void buildDNodes(DOctTree **h_nodes,
        DTriangle ***d_triangles, int *hierarchy, int *child_pos,
        int &index, int father_index, int pos) const = 0;
    virtual ~OctNode(){}
};

class OctBranch: public OctNode {

  private:
    std::vector<OctNode*> _children;

  public:
    OctBranch(const std::vector<float>&ini,
        const std::vector<float>&end);
    void insert(Object *obj);
    void upgrade(int slot, std::vector<Object*> objs,
        const std::vector<float>&ini,
        const std::vector<float>&end);
    virtual bool intersectsChildren(const std::vector<float> &e,
      const std::vector<float> &d, std::vector<float> &iP,
      std::vector<float> &n, float &t,
      std::vector<const Material*>&mats) {

      bool inter = false;
      for (auto &c: _children)
        inter = c->intersectsRay(e,d,iP,n,t,mats) || inter;
      return inter;
    }
    virtual ~OctBranch() {
      for (auto &c: _children) delete c;
    }
    void countNode(int &count) const override;
    void buildDNodes(DOctTree **h_nodes, DTriangle ***d_triangles,
        int *hierarchy, int *child_pos, int &index,
        int father_index, int pos) const override;
};


class OctLeave: public OctNode {

  private:
    std::vector<Object*> _objs; int _slot; OctBranch *_father;

  public:
    OctLeave(int, const std::vector<float>&,
        const std::vector<float>&,
        OctBranch *f);
    void insert(Object *obj);
    virtual bool intersectsChildren(const std::vector<float> &e,
      const std::vector<float> &d, std::vector<float> &iP,
      std::vector<float> &n, float &t,
      std::vector<const Material*>&mats) {

      bool inter = false;
      for (auto &obj: _objs)
        inter = obj->intersectsRay(e,d,iP,n,t,mats) || inter;
      return inter;
    }
    void countNode(int &count) const override;
    void buildDNodes(DOctTree **h_nodes, DTriangle ***d_triangles,
        int *hierarchy, int *child_pos, int &index,
        int father_index, int pos) const override;
};

class Mesh: public Object {

  private:
    std::vector<Triangle*> _t; std::vector<Vertex*> _v;

  public:
    Mesh(const std::pair<std::vector<Triangle*>,
        std::vector<Vertex*>>&md,
        const std::vector<const Material*>&mats): 
      _t(md.first),_v(md.second) { _mats = mats;
      for (auto &t: _t) t->_mesh = this; }
    bool intersectsRay(const std::vector<float>&e,
        const std::vector<float>&d,
        std::vector<float>&iPoint, std::vector<float>&n, float&t,
        std::vector<const Material*>&mats) override;
    virtual bool intersectsOctNode(const std::vector<float>&ini,
        const std::vector<float>&end){return false;}
    void insertInto(OctNode *n);
    void countTriangles(int &count) { count += _t.size(); }
    void buildDTriangles(DTriangle **htod_triangles,
       DMaterial **mats, int nMats, int &index);
    void buildDMats(DMaterial ***htod_mats, DMaterial ***htod_mat,
        int *nMats);
    void scale(const std::vector<float> &sca);
    void rotate(const std::vector<float> &rot);
    void translate(const std::vector<float> &trans);
    ~Mesh(); 
};

class Sphere: public Object {

  private:
    float _r; std::vector<float> _c;

  public:
    Sphere(float r,const std::vector<float>&c,
        std::vector<const Material*>&mats): _r(r), _c(c)
      { _mats = mats; }
    virtual ~Sphere(void){};
    bool intersectsRay(const std::vector<float>&e,
        const std::vector<float>&d,
        std::vector<float>&iSphere, std::vector<float>&n, float&t,
        std::vector<const Material*>&mats) override;
    virtual bool intersectsOctNode(const std::vector<float>&ini,
        const std::vector<float>&end){ return false; }
    const std::vector<float>& getCenter(void) const { return _c; }
    float getRadius(void) { return _r; }
};

class Ray {

  private:

    std::vector<float> &_color;
    const std::vector<float> _e,_d;
    std::stack<float> _level;
    unsigned char _depth;

  public:

    Ray(std::vector<float> &color, const std::vector<float> &e,
        const std::vector<float> &d, std::stack<float> &level,
        unsigned char depth):
      _color(color),_e(e),_d(d),_level(level),_depth(depth) {}
    bool intersect(void);
};

class Camera {

  private:

    unsigned short _width,_height;
    std::vector<float> _e,_up,_target;
    float _fov,_near;
    

  public:

    Camera(unsigned short W, unsigned short H, float fov,
        std::vector<float> &e, std::vector<float> &up, 
        std::vector<float> &target): _width(W),_height(H),_e(e),
    _up(up),_target(target),_fov(fov),_near(NEAR) {}
    void film(std::string img_name);
    DCamera* buildDCamera(void) const;
};


class CameraGPUAdapter {

  private:

    DCamera *d_c;

  public:

    CameraGPUAdapter(const Camera *c);
    ~CameraGPUAdapter(void);
    DCamera* getDCamera(void) const { return d_c; }
};


class LightsGPUAdapter {

  private:

    DLights *d_l, *h_l; DLight **htod_l;
    int _nLights;

  public:

    LightsGPUAdapter(const std::vector<Light*> &l);
    ~LightsGPUAdapter(void);
    DLights* getDLights(void) const { return d_l; }
};


class SphereGPUAdapter {

  private:

    DSphere *h_s,*d_s;
    DMaterial **d_mats, **htod_mats;

  public:

    SphereGPUAdapter(Sphere *s);
    DSphere* getDS(void) const {return d_s;}
    ~SphereGPUAdapter(void);
};

class OctGPUAdapter {

  private:

    DOctTree **htod_nodes;
    DTriangle ***htod_nodeTriangles; int _count;

  public:

    OctGPUAdapter(const OctNode *o);
    DOctTree* getDOct(void) const {
      if (htod_nodes) return htod_nodes[0];
      else return NULL;
    }
    ~OctGPUAdapter(void);
};

class MeshGPUAdapter {

  private:
    
    DTriangle **htod_triangles;
    DMaterial ***htod_mats, ***htod_mat;
    int _count,_meshes, *nMats;

  public:

    MeshGPUAdapter(std::vector<Mesh*> meshes);
    ~MeshGPUAdapter(void);
};


// -> Objects Adapter

class ObjectsGPUAdapter {
  
  private:
    
    DSpheres *d_sphs, *h_sphs;
    std::vector<SphereGPUAdapter*> obj_adapters;

  public:

    ObjectsGPUAdapter(const std::vector<Object*> &objects);
    ~ObjectsGPUAdapter(void);
    DSpheres* getDObjects(void) const { return d_sphs; }
};




extern std::vector<Object*> allObjects;
extern std::vector<Mesh*> allMeshes;
extern std::vector<Light*> allLights;
extern std::map<std::string,Material*> allMaterials;
extern std::map<Triangle*,DTriangle*> t_map;
extern std::vector<float> BACKGROUND_COLOR;
extern unsigned short W,H,DEPTH;
extern float SCENE_REFR;
extern bool SHADOWS_ENABLED;

static float INF = 64;


#endif /*MODEL_H*/
