//#include <iostream>
//#include <vector>
//#include <omp.h>
//using namespace std;
//
//// Parallel LU decomposition
//void parallelLU(vector<vector<double>>& A,
//    vector<vector<double>>& L,
//    vector<vector<double>>& U,
//    int n)
//{
//    for (int k = 0; k < n; k++) {
//
//        // Compute U row (single-thread)
//        for (int j = k; j < n; j++) {
//            double sum = 0;
//            for (int p = 0; p < k; p++)
//                sum += L[k][p] * U[p][j];
//            U[k][j] = A[k][j] - sum;
//        }
//
//        // Compute L column (parallel)
//#pragma omp parallel for
//        for (int i = k; i < n; i++) {
//            if (i == k) L[i][k] = 1;
//            else {
//                double sum = 0;
//                for (int p = 0; p < k; p++)
//                    sum += L[i][p] * U[p][k];
//
//                L[i][k] = (A[i][k] - sum) / U[k][k];
//            }
//        }
//    }
//}
//
//vector<double> forwardSolve(const vector<vector<double>>& L,
//    const vector<double>& b,
//    int n)
//{
//    vector<double> y(n);
//    for (int i = 0; i < n; i++) {
//        double sum = 0;
//        for (int j = 0; j < i; j++)
//            sum += L[i][j] * y[j];
//        y[i] = b[i] - sum;
//    }
//    return y;
//}
//
//vector<double> backwardSolve(const vector<vector<double>>& U,
//    const vector<double>& y,
//    int n)
//{
//    vector<double> x(n);
//    for (int i = n - 1; i >= 0; i--) {
//        double sum = 0;
//        for (int j = i + 1; j < n; j++)
//            sum += U[i][j] * x[j];
//        x[i] = (y[i] - sum) / U[i][i];
//    }
//    return x;
//}
//
//int main() {
//    int n = 3;
//
//    vector<vector<double>> A = {
//        {2, 3, 1},
//        {4, 7, 7},
//        {6, 18, 22}
//    };
//
//    vector<double> b = { 1, 2, 3 };
//
//    vector<vector<double>> L(n, vector<double>(n, 0));
//    vector<vector<double>> U(n, vector<double>(n, 0));
//
//    double start = omp_get_wtime();
//
//    parallelLU(A, L, U, n);
//    vector<double> y = forwardSolve(L, b, n);
//    vector<double> x = backwardSolve(U, y, n);
//
//    double end = omp_get_wtime();
//
//    cout << "OpenMP LU Solution:\n";
//    for (int i = 0; i < n; i++)
//        cout << "x[" << i << "] = " << x[i] << endl;
//
//    double parallel_time = end - start;
//    cout << "\nParallel Execution Time: " << parallel_time << " seconds\n";
//
//    return 0;
//}
