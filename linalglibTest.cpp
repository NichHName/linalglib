#include <iostream>
#include "linalglib.cpp"
#include <vector>
#include <complex>
#include <stdexcept>

int main() {
    // Declarations
    linalglib::Matrix<double> A(2, 3); // Testing this way of defining a matrix
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3;
    A(1, 0) = 4.0;
    A(1, 1) = 5.0;
    A(1, 2) = 6.0;
    std::vector<double> matrixb = {5, 3, -10, 7, 2, 1}; // Testing the way better of making a matrix
    linalglib::Matrix B(matrixb, 3, 2);

    std::vector<double> matrixc = {17.0, 117.0, 2005.0, 2004.0};
    linalglib::Matrix C(matrixc, 2, 2);

    // Complex matrix
    std::vector<std::complex<double>> matrixd = { {1.0, 2}, {3, 5}, {4, -2}, {6, 0} };
    linalglib::Matrix D(matrixd, 2, 2);

    std::vector<double> v1 = {1.0, 6.0, -3.0};
    std::vector<double> v2 = {4.0, 0.0, 7.0};

    // 1. Test matrix construction and element access
    std::cout << "Matrix A:" << std::endl;
    displayMatrix(A);

    // 2. Test inner product
    double ip = linalglib::innerProduct(v1, v2);
    std::cout << "Inner product of v1 and v2 (should be -17): " << ip << std::endl;

    // 3. Test matrix-vector multiplication
    std::vector<double> mv_result1 = linalglib::matvec(A, v1);
    std::cout << "Matrix-vector product A * v1: [" << mv_result1[0] << "  " << mv_result1[1] << "]" << std::endl;

    std::vector<double> mv_result2 = linalglib::matvec(A, v2);
    std::cout << "Matrix-vector product A * v2: [" << mv_result2[0] << "  " << mv_result2[1] << "]" << std::endl;

    // 4. Test matrix-matrix multiplication
    linalglib::Matrix AB = linalglib::matmul(A, B);
    std::cout << "Matrix-matrix product AB: " << std::endl;
    displayMatrix(AB);

    linalglib::Matrix BA = linalglib::matmul(B, A);
    std::cout << "Matrix-matrix product BA: " << std::endl;
    displayMatrix(BA);

    linalglib::Matrix BC = linalglib::matmul(B, C);
    std::cout << "Matrix-matrix product BE: " << std::endl;
    displayMatrix(BC);

    linalglib::Matrix At = linalglib::transpose(A);
    std::cout << "Transpose of A: " << std::endl;
    displayMatrix(At);

    // Complex Testing
    std::cout << "Complex matrix D:" << std::endl;
    displayMatrix(D);

    // G transpose, G conjugate transpose
    linalglib:: Matrix Dt = linalglib::transpose(D);
    linalglib:: Matrix Dct = linalglib::conjugateTranspose(D);

    std::cout << "Regular transpose of D:" << std::endl;
    displayMatrix(Dt);

    std::cout << "Conjugate transpose of D:" << std::endl;
    displayMatrix(Dct);

    // Test findEigen
    std::pair<std::vector<double>, linalglib::Matrix<double>> eigenResult = findEigen(C);

    std::vector<double> eigenvalC = eigenResult.first;

    std::cout << "Eigenvalues of C:" << std::endl;
    linalglib::displayVector(eigenvalC);
}