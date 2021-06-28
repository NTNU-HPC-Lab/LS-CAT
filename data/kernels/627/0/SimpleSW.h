#ifndef SIMPLESW_CUH
#define SIMPLESW_CUH

class Data;

class SimpleSW{
public:
	SimpleSW(const Data& txt, const Data& ptn, int threshold = 0xffff);
	~SimpleSW();
private:
	void call_DP(const Data& txt, const Data& ptn);
	void checkScore(const char* direction, const int* score, const Data& txt) const;
	void traceback(const char* direction, const Data& txt, int txt_point, int ptn_point) const; 
	void show(const char* score, const Data& txt, const Data& ptn) const;
};


#endif





