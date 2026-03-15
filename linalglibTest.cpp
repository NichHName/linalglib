#include <iostream>
#include "linalglib.cpp"
#include <vector>
#include <complex>
#include <stdexcept>
#include <type_traits>

using namespace linalglib;

int main() {
    //----------------------------------------------------------------
    // Matrix/Vector Creation
    //----------------------------------------------------------------
    
    Matrix<double> A(2, 3); // Testing this way of defining a matrix
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3;
    A(1, 0) = 4.0;
    A(1, 1) = 5.0;
    A(1, 2) = 6.0;
    std::vector<double> matrixb = {5, 3, -10, 7, 2, 1}; // Testing the way better of making a matrix
    Matrix B(matrixb, 3, 2);

    std::vector<double> matrixc = {17.0, 117.0, 2005.0, 2004.0};
    Matrix C(matrixc, 2, 2);

    // Complex matrix
    std::vector<std::complex<double>> matrixd = { {1.0, 2}, {3, 5}, {4, -2}, {6, 0} };
    Matrix D(matrixd, 2, 2);

    std::vector<double> v1 = {1.0, 6.0, -3.0};
    std::vector<double> v2 = {4.0, 0.0, 7.0};

    //----------------------------------------------------------------
    // Test matrix construction and element access
    //----------------------------------------------------------------

    std::cout << "Matrix A:" << std::endl;
    displayMatrix(A);

    //----------------------------------------------------------------
    // Test vecAdd, matAdd
    //----------------------------------------------------------------

    std::vector<double> v3 = vecAdd(v1, v2);
    std::cout << "Sum of v1 and v2:" << std::endl;
    displayVector(v3);

    Matrix CpD = matAdd(C, D);
    std::cout << "Sum of matrices C and D:" << std::endl;
    displayMatrix(CpD);

    //----------------------------------------------------------------
    // Test inner product
    //----------------------------------------------------------------

    double ip = innerProduct(v1, v2);
    std::cout << "Inner product of v1 and v2 (should be -17): " << ip << std::endl;

    //----------------------------------------------------------------
    // Test matrix-vector multiplication
    //----------------------------------------------------------------

    std::vector<double> mv_result1 = matvec(A, v1);
    std::cout << "Matrix-vector product A * v1: [" << mv_result1[0] << "  " << mv_result1[1] << "]" << std::endl;

    std::vector<double> mv_result2 = matvec(A, v2);
    std::cout << "Matrix-vector product A * v2: [" << mv_result2[0] << "  " << mv_result2[1] << "]" << std::endl;

    //----------------------------------------------------------------
    // Test matrix-matrix multiplication
    //----------------------------------------------------------------

    Matrix AB = matmul(A, B);
    std::cout << "Matrix-matrix product AB: " << std::endl;
    displayMatrix(AB);

    Matrix BA = matmul(B, A);
    std::cout << "Matrix-matrix product BA: " << std::endl;
    displayMatrix(BA);

    Matrix BC = matmul(B, C);
    std::cout << "Matrix-matrix product BE: " << std::endl;
    displayMatrix(BC);

    Matrix At = transpose(A);
    std::cout << "Transpose of A: " << std::endl;
    displayMatrix(At);

    //----------------------------------------------------------------
    // Complex Testing
    //----------------------------------------------------------------

    std::cout << "Complex matrix D:" << std::endl;
    displayMatrix(D);

    // G transpose, G conjugate transpose
    Matrix Dt = transpose(D);
    Matrix Dct = conjugateTranspose(D);

    std::cout << "Regular transpose of D:" << std::endl;
    displayMatrix(Dt);

    std::cout << "Conjugate transpose of D:" << std::endl;
    displayMatrix(Dct);

    //----------------------------------------------------------------
    // Test findEigen
    //----------------------------------------------------------------

    std::pair<std::vector<double>, Matrix<double>> eigenResult = findEigen(C);

    std::vector<double> eigenvalC = eigenResult.first;

    std::cout << "Eigenvalues of C:" << std::endl;
    displayVector(eigenvalC);

    //----------------------------------------------------------------
    // Test svd, svdTruncated
    //----------------------------------------------------------------

    std::cout << "Original Matrix C:" << std::endl;
    displayMatrix(C);

    auto [Uc, Sc, Vtc] = svd(C);

    std::cout << "U:" << std::endl;
    displayMatrix(Uc);
    std::cout << "S (Sigma):" << std::endl;
    displayMatrix(Sc);
    std::cout << "V^T:" << std::endl;
    displayMatrix(Vtc);

    std::cout << "Reconstructed Matrix C (U * S * V^T):" << std::endl;
    Matrix USc = matmul(Uc, Sc);
    Matrix USVtc = matmul(USc, Vtc);
    displayMatrix(USVtc);


    std::cout << "Original Matrix A:" << std::endl;
    displayMatrix(A);

    auto [Ua, Sa, Vta] = svd(A);

    std::cout << "U:" << std::endl;
    displayMatrix(Ua);
    std::cout << "S (Sigma):" << std::endl;
    displayMatrix(Sa);
    std::cout << "V^T:" << std::endl;
    displayMatrix(Vta);

    std::cout << "Reconstructed Matrix A (U * S * V^T):" << std::endl;
    Matrix USa = matmul(Ua, Sa);
    Matrix USVta = matmul(USa, Vta);
    displayMatrix(USVta);


    std::cout << "Original Matrix C:" << std::endl;
    displayMatrix(C);

    auto [Uct, Sct, Vtct] = svdTruncated(C, 1);

    std::cout << "Truncated U (k=1):" << std::endl;
    displayMatrix(Uct);
    std::cout << "Truncated S (k=1):" << std::endl;
    displayMatrix(Sct);
    std::cout << "Truncated V^T (k=1):" << std::endl;
    displayMatrix(Vtct);

    std::cout << "Rank-1 Approximation of Matrix C (U * S * V^T):" << std::endl;
    Matrix USct = matmul(Uct, Sct);
    Matrix USVtct = matmul(USct, Vtct);
    displayMatrix(USVtct);


    std::cout << "Original Matrix B:" << std::endl;
    displayMatrix(B);

    auto [Ubt, Sbt, Vtbt] = svdTruncated(B, 1);

    std::cout << "Truncated U (k=1):" << std::endl;
    displayMatrix(Ubt);
    std::cout << "Truncated S (k=1):" << std::endl;
    displayMatrix(Sbt);
    std::cout << "Truncated V^T (k=1):" << std::endl;
    displayMatrix(Vtbt);

    std::cout << "Rank-1 Approximation of Matrix B (U * S * V^T):" << std::endl;
    Matrix USbt = matmul(Ubt, Sbt);
    Matrix USVtbt = matmul(USbt, Vtbt);
    displayMatrix(USVtbt);
}