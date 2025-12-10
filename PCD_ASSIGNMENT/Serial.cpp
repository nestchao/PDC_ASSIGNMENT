#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>  
using namespace std;

void calculateLU(vector<vector<double>>& A,
    vector<vector<double>>& L,
    vector<vector<double>>& U,
    int size) {

    for (int k = 0; k < size; k++) {
        for (int j = k; j < size; j++) {
            double sum = 0;
            for (int p = 0; p < k; p++) {
                sum += L[k][p] * U[p][j];
            }
            U[k][j] = A[k][j] - sum;
        }

        for (int i = k; i < size; i++) {
            if (i == k) {
                L[i][k] = 1;
            }
            else {
                double sum = 0;
                for (int p = 0; p < k; p++) {
                    sum += L[i][p] * U[p][k];
                }
                L[i][k] = (A[i][k] - sum) / U[k][k];
            }
        }
    }
}

vector<double> forwardSolve(const vector<vector<double>>& L, const vector<double>& b, int size) {
    vector<double> y(size);
    for (int i = 0; i < size; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = b[i] - sum;
    }
    return y;
}

vector<double> backwardSolve(const vector<vector<double>>& U, const vector<double>& y, int n) {
    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++)
            sum += U[i][j] * x[j];
        x[i] = (y[i] - sum) / U[i][i];
    }
    return x;
}

int main() {

    vector<vector<double>> A = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 10.0},
        {7.0, 8.0, 9.0}
    };

    vector<double> B = { 10, 6, 8 };

    int size = A.size();

    vector<vector<double>> L(size, vector<double>(size, 0));
    vector<vector<double>> U(size, vector<double>(size, 0));


    cout << "Starting Serial Calculation..." << endl;

    double start_time = omp_get_wtime();

    calculateLU(A, L, U, size);

    vector<double> y = forwardSolve(L, B, size);
    vector<double> x = backwardSolve(U, y, size);

    double end_time = omp_get_wtime();

    cout << "Calculation Complete." << endl;
    cout << "-----------------------------------" << endl;
    cout << "Matrix Size: " << size << endl;
    cout << "Execution Time: " << (end_time - start_time) << " seconds" << endl;
    cout << "-----------------------------------" << endl;

    cout << "Results of X:" << endl;
    for (int i = 0; i < x.size(); i++) {
        cout << "x[" << i << "] = " << x[i] << endl;
    }

    return 0;
}