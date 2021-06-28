#pragma once
#include <vector>
#include <iostream>
#include <numeric>

template <class T>
class Matrix {
public:
	struct SizeInfo {
		size_t n_rows;
		size_t n_cols;

		bool operator==(const SizeInfo&) const;
		SizeInfo Swap() const;
		size_t TotalSize() const;
	};

	Matrix() = default;
	Matrix(size_t, size_t, std::enable_if_t<std::is_trivially_constructible_v<T>>* = 0);
	Matrix(size_t, size_t, const T&);
	Matrix(size_t, size_t, T*);
	Matrix(const SizeInfo&, std::enable_if_t<std::is_trivially_constructible_v<T>>* = 0);
	Matrix(const SizeInfo&, const T&);
	Matrix(const SizeInfo&, T*);

	Matrix(const std::vector<std::vector<T>>&);
	Matrix(const Matrix&) = default;
	Matrix(Matrix&&) = default;
	~Matrix() = default;

	Matrix& operator=(const Matrix&) = default;
	Matrix& operator=(Matrix&&) = default;

	std::vector<T>& operator[](size_t row);
	const std::vector<T>& operator[](size_t row) const;

	bool operator==(const Matrix&) const;

	SizeInfo Size() const;
	size_t Rows() const;
	size_t Cols() const;

	Matrix& Transpose();
	Matrix Transposed() const;

	void Pad(size_t thickness);
	Matrix Padded(size_t thickness) const;

	bool CanMultiply(const Matrix& multiplier) const;
	Matrix operator*(const Matrix& multiplier) const;
	Matrix LazyMultiply(const Matrix& multiplier) const;

	const std::vector<std::vector<T>>& RawData() const;

	std::vector<T> Flatten() const;

	template <class T>
	friend std::ostream& operator<<(std::ostream& out, const Matrix<T>& matrix) {
		for (size_t row = 0; row < matrix.size_.n_rows; ++row) {
			for (size_t col = 0; col < matrix.size_.n_cols - 1; ++col) {
				out << matrix.data_[row][col] << ' ';
			}
			out << matrix.data_[row][matrix.size_.n_cols - 1] << '\n';
		}
		return out;
	}

	template <class T>
	friend std::istream& operator>>(std::istream& in, Matrix<T>& matrix) {
		for (size_t row = 0; row < matrix.size_.n_rows; ++row) {
			for (size_t col = 0; col < matrix.size_.n_cols; ++col) {
				in >> matrix.data_[row][col];
			}
		}
		return in;
	}

private:
	std::vector<std::vector<T>> data_;
	SizeInfo size_;
};

template <class T>
bool Matrix<T>::SizeInfo::operator==(const SizeInfo& other) const {
	return (n_rows == other.n_rows) && (n_cols == other.n_cols);
}

template <class T>
typename Matrix<T>::SizeInfo Matrix<T>::SizeInfo::Swap() const {
	SizeInfo result;
	result.n_rows = n_cols;
	result.n_cols = n_rows;
	return result;
}

template <class T>
size_t Matrix<T>::SizeInfo::TotalSize() const {
	return n_rows * n_cols;
}

template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols,
				  std::enable_if_t<std::is_trivially_constructible_v<T>>*) {
	size_.n_rows = rows;
	size_.n_cols = cols;
	data_ = std::vector<std::vector<T>>(rows, std::vector<T>(cols));
}

template <class T>
Matrix<T>::Matrix(const SizeInfo& size, std::enable_if_t<std::is_trivially_constructible_v<T>>*)
	: Matrix(size.n_rows, size.n_cols) {
}

template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols, T* values) : Matrix(rows, cols) {
	for (size_t row = 0; row < rows; ++row) {
		for (size_t col = 0; col < cols; ++col) {
			size_t idx = row * cols + col;
			data_[row][col] = values[idx];
		}
	}
}

template <class T>
Matrix<T>::Matrix(const SizeInfo& size, T* values) : Matrix(size) {
	for (size_t row = 0; row < size.n_rows; ++row) {
		for (size_t col = 0; col < size.n_cols; ++col) {
			size_t idx = row * size.n_cols + col;
			data_[row][col] = values[idx];
		}
	}
}

template <class T>
std::vector<T>& Matrix<T>::operator[](size_t row) {
	return data_[row];
}

template <class T>
const std::vector<T>& Matrix<T>::operator[](size_t row) const {
	return data_[row];
}

template <class T>
bool Matrix<T>::operator==(const Matrix<T>& other) const {
	if (size_ == other.size_) {
		return false;
	}
	for (size_t i = 0; i < n_rows; ++i) {
		for (size_t j = 0; j < n_cols; ++j) {
			if (data_[i][j] != other.data_[i][j]) {
				return false;
			}
		}
	}
	return true;
}

template <class T>
bool Matrix<T>::CanMultiply(const Matrix<T>& other) const {
	return size_.n_cols == other.size_.n_rows;
}

template <class T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& multiplier) const {
	if (!CanMultiply(multiplier)) {
		throw std::invalid_argument("Matrix sizes do not agree");
	}
	Matrix<T> result(size_.n_rows, multiplier.size_.n_cols);
	for (size_t i = 0; i < size_.n_rows; ++i) {
		for (size_t j = 0; j < multiplier.size_.n_cols; ++j) {
			for (size_t k = 0; k < size_.n_cols; ++k) {
				result[i][j] += data_[i][k] * multiplier.data_[k][j];
			}
		}
	}
	return result;
}

template <class T>
Matrix<T> Matrix<T>::LazyMultiply(const Matrix<T>& multiplier) const {
	if (!CanMultiply(multiplier)) {
		throw std::invalid_argument("Matrix sizes do not agree");
	}
	Matrix<T> transpose = multiplier.Transposed();
	Matrix<T> result(size_.n_rows, multiplier.size_.n_cols);
	for (size_t i = 0; i < size_.n_rows; ++i) {
		for (size_t j = 0; j < multiplier.size_.n_cols; ++j) {
			result[i][j] = std::inner_product(data_[i].begin(), data_[i].end(),
											  transpose.data_[j].begin(), T());
		}
	}
	return result;
}

template <class T>
typename Matrix<T>::SizeInfo Matrix<T>::Size() const {
	return size_;
}

template <class T>
size_t Matrix<T>::Rows() const {
	return size_.n_rows;
}

template <class T>
size_t Matrix<T>::Cols() const {
	return size_.n_cols;
}

template <class T>
void Matrix<T>::Pad(size_t thickness) {
	std::vector<std::vector<T>> old_data = data_;
	data_ = std::vector<std::vector<T>>(old_data.size() + 2 * thickness, std::vector<T>(old_data[0].size() + 2 * thickness, T()));
	for (size_t row = 0; row < old_data.size(); ++row) {
		std::copy(old_data[row].begin(), old_data[row].end(), data_[thickness + row].begin() + thickness);
	}
	size_.n_rows += 2 * thickness;
	size_.n_cols += 2 * thickness;
}

template <class T>
Matrix<T> Matrix<T>::Padded(size_t thickness) const {
	Matrix<T> padded(*this);
	padded.Pad(thickness);
	return padded;
}

template <class T>
Matrix<T>& Matrix<T>::Transpose() {
	if (size_.n_rows == size_.n_cols) {
		for (size_t i = 0; i < n_rows; ++i) {
			for (size_t j = i + 1; j < n_cols; ++j) {
				std::swap(data_[i][j], data_[j][i]);
			}
		}
	} else {
		*this = Transposed();
	}
	return *this;
}

template <class T>
Matrix<T> Matrix<T>::Transposed() const {
	Matrix<T> transposed(size_.Swap());
	for (size_t i = 0; i < size_.n_rows; ++i) {
		for (size_t j = 0; j < size_.n_cols; ++j) {
			transposed.data_[j][i] = data_[i][j];
		}
	}
	return transposed;
}

template <class T>
std::vector<T> Matrix<T>::Flatten() const {
	std::vector<T> result;
	result.reserve(std::accumulate(data_.begin(), data_.end(), 0ULL,
				  [](size_t sum, std::vector<T> v) -> size_t {
					  return sum + v.size();
				  }));
	for (size_t row = 0; row < size_.n_rows; ++row) {
		std::copy(data_[row].begin(), data_[row].end(), std::back_inserter(result));
	}
	return result;
}
