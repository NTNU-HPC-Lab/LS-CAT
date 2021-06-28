

	int multipy_1d(
		float  *a,
		float  *b,
		float  *c,
		const int matrix_size
	);

	int multipy_1d_diff_dim(
		float  *a,
		float  *b,
		float  *c,
		const int m,
		const int n, 
		const int k
	);

	int alocate_1d(
		float** a, 
		const int matrix_size
	);

	void set_identity_1d(
		float* a, 
		const int matrix_size
	);

	int multipy_1d_only_gpu(
		float  *a,
		float  *b,
		float  *c,
		const int matrix_size
	);

	int multipy_1d_diff_dim_only_gpu(
		float  *a,
		float  *b,
		float  *c,
		const int m,
		const int n, 
		const int k
	);
