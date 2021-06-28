#pragma once
#include "Object.h"

class Transform : public Object
{
	static std::vector<Transform*> movedTransform;

	friend class GameObject;
	friend class SceneGraph;
private:
	union {
		struct {
			glm::vec3 right;
			glm::vec3 up;
			glm::vec3 forward;
		};

		struct {
			glm::vec3 axis[3];
		};
	};

private:
	std::vector<Transform*> v_Children;

	GetMacro(Transform*, Parent, parent);
	GetMacro(GameObject*, GameObject, gameObject);

private:
	glm::vec3 worldRightAxis;
	glm::vec3 worldUpAxis;
	glm::vec3 worldForwardAxis;

	glm::vec3 worldPosition;
	glm::vec3 lossyScale;

	glm::vec3 localPosition;
	glm::vec3 localScale;

	glm::mat4x4 worldMatrix;
	glm::mat4x4 localMatrix;

	bool noRecalculateChild;
	bool moved;
public:
	Transform(bool noRecalculateChild = false);
	virtual ~Transform();

public:

	//Translate
	Transform& SetWorldPosition(float x, float y, float z);
	Transform& SetWorldPosition(glm::vec3 pos);

	Transform& AddWorldPosition(float dx, float dy, float dz);
	Transform& AddWorldPosition(glm::vec3 delta);

	Transform& SetLocalPosition(float x, float y, float z);
	Transform& SetLocalPosition(glm::vec3 pos);

	Transform& AddLocalPosition(float dx, float dy, float dz);
	Transform& AddLocalPosition(glm::vec3 delta);

	//Scale
	Transform& SetLocalScale(float x, float y, float z);
	Transform& SetLocalScale(glm::vec3 scale);

	//Rotate
	Transform& SetRotateLocal(const glm::quat& quaternion);
	Transform& SetRotateLocal(const glm::mat3x3& matRotation);

	Transform& SetRotateWorld(const glm::quat& quaternion);
	Transform& SetRotateWorld(const glm::mat3x3& matRotation);

	Transform& SetRotateAxisX(const glm::vec3& axis);
	Transform& SetRotateAxisY(const glm::vec3& axis);
	Transform& SetRotateAxisZ(const glm::vec3& axis);

	Transform& RotateAxisWorld(float axisXAngle, float axisYAngle, float axisZAngle);
	Transform& RotateAxisWorld(glm::vec3 deltaAngle);

	Transform& RotateAxisLocal(float axisXAngle, float axisYAngle, float axisZAngle);
	Transform& RotateAxisLocal(glm::vec3 deltaAngle);

	//특정 방향을 바라보게 회전해라.
	Transform& LookDirection(const glm::vec3& dir, const glm::vec3& Up = glm::vec3(0, 1, 0));

	//특정방향을 바라보는데 angle 각만큼만 회전 해라
	Transform& LookDirection(const glm::vec3& dir, float angle);

	//특정위치를 바라보게 회전해라.
	Transform& LookPosition(const glm::vec3& pos, const glm::vec3& Up = glm::vec3(0, 1, 0));

	//특정위치를  바라보는데 angle 각만큼만 회전 해라
	Transform& LookPosition(const glm::vec3& pos, float angle);

	//Transform 을 업데이트 한다 ( Trasform 의 정보가 갱신되었을때 사용된다 )
	Transform& UpdateTransform();

	// ------------------------------------------------------------------------
	// Get 관련
	// ------------------------------------------------------------------------

	//월드 위치를 얻는다.
	glm::vec3 GetWorldPosition() const;
	glm::vec3 GetLocalPosition() const;

	//크기를 얻는다.
	glm::vec3 GetLocalScale()const;
	glm::vec3 GetLossyScale()const;

	//최종행렬을 얻는다.
	glm::mat4x4 GetWorldMatrix() const;
	glm::mat4x4 GetLocalMatrix() const;
	Transform& SetWorldMatrix(glm::mat4x4 mat);
	Transform& SetLocalMatrix(glm::mat4x4 mat);

	//축을 얻는다. 
	glm::vec3 GetForward(bool bNormalize = true) const;
	glm::vec3 GetUp(bool bNormalize = true) const;
	glm::vec3 GetRight(bool bNormalize = true) const;
	void GetScaledAxies(glm::vec3* pVecArr) const;
	void GetUnitAxies(glm::vec3* pVecArr) const;
	glm::vec3 GetScaledAxis(int axisNum) const;
	glm::vec3 GetUnitAxis(int axisNum) const;

	//월드 행렬에서 회전행렬 성분만 가져온다.
	glm::mat3x3 GetWorldRotateMatrix() const;
	glm::mat3x3 GetLocalRotateMatrix() const;

	//월드사원수를 얻는다.
	glm::quat GetWorldQuaternion() const;

	Transform& operator*=(const Transform& trans);
	Transform& operator*=(const glm::mat4x4& matrix);

	//SetParent를 중점으로 구현
	Transform& SetParent(Transform* parent, bool attachWorld = true);
	Transform& AddChild(Transform* child, bool attachWorld = true);

	Transform* GetChild(int index) { return v_Children[index]; }
	size_t GetChildCount() { return v_Children.size(); }

	Transform* GetRoot();

	Transform* GetChildFromName(std::string name);

private:
	void RecalcuateBoundingSphere();
};