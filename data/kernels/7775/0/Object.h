#pragma once
class Object
{
public:
	Object();
	Object(std::string name);
	virtual ~Object();

	virtual std::string ToString();
	
	GetSetMacro(std::string, Name, name)
};