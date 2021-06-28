#ifndef DATA_H
#define DATA_H

// 塩基配列のデータ
class Data{
public:
	// ファイル読み込み
	Data(const char* fname);
	// ランダム生成
	Data(int num);
	char operator[](int i) const;
	int size() const;
	const char* data() const;
	~Data();
private:
	char* mData;
	int mSize;	
	enum Acid{
		A = 0,
		G = 1,
		C = 2,
		T = 3
	};
};


#endif
