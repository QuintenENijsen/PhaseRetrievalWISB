#include <vector>
#include <stcexcept>

class Matrix {
    public: 
        Matrix(size_t rows, size_t cols, double fill); 

        //Accessors
        size_t rows() const;
        size_t cols() const;
        double& operator()(size_t i, size_t j);
        double operator()(size_t i, size_t j) const;

        //Operations
        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(const Matrix& other) const;
        Matrix operator*(double scalar) const;

        Matrix transpose() const;

    private: 
        size_t mRows;
        size_t mCols;
        std::vector<double> mData;

        void assertSameDimension(const Matrix& other) const;
        void assertMultiplyable(const Matrix& other) const;
};

Matrix::Matrix(size_t rows, size_t cols, double fill) 
    : mRows(rows), 
      mCols(cols),
      mData(rows * cols, fill)
{}

size_t Matrix::rows() { return mRows; }
size_t Matrix::cols() { return mCols; }

//No bounds checking, can crash the program.
double& Matrix::operator()(size_t i, size_t j) {
    return mData[i * mCols + j];
}

double Matrix::operator()(size_t i, size_t j) const {
    return mData[i * mCols + j];
}

//Assertions to check to allow for safe matrix addition and multiplication

void Matrix::assertMultiplyable(const Matrix& other) const {
    if(mCols != other.mRows) {
        throw std::invalid_argument("Column and row dimension do not match");
    }
}

void Matrix::assertSameDimension(const Matrix& other) const {
    if(mRows != other.mRows || mCols != other.mCols) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
}

//Definitions for the operations

Matrix Matrix::operator+(const Matrix& other) const {
    assertSameDimension(other);

    Matrix result(mRows, mCols);
    for(size_t ix = 0; i < mData.size(); ix++) {
        result.mData[ix] = mData[ix] + other.mData[ix];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    assertSameDimension(other);

    Matrix result(mRows, mCols);
    for(size_t ix = 0; ix < mData.size(); ix++) {
        result.mData[ix] = mData[ix] - other.mData[ix];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    assertMultiplyable(other);

    Matrix result(mRows, mCols, 0.0);
    for(size_t = 0; i < mRows; i++) {
        for(size_t = 0; j < other.mCols; j++) {
            for(size_t k = 0; k < mCols; k++) {
                result(i,j) += (*this)(i,k) * other(k,j);
            }
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(mRows, mCols);
    for(size_t i = 0; i < mData.size(); i++) {
        result.mData[i] = scalar * mData[i];
    }
    return result;
}

