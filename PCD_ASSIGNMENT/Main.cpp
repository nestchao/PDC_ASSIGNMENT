#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <cstdlib>

using namespace std;

void initMatrix(int n, vector<vector<double>>& A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (rand() % 10) + 1;
        }
        A[i][i] += n;
    }
}

void resetLU(int n, vector<vector<double>>& L, vector<vector<double>>& U) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = 0.0;
            U[i][j] = 0.0;
            if (i == j) L[i][i] = 1.0;
        }
    }
}

void lu_serial(int n, const vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U) {
    for (int i = 0; i < n; i++) {
        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[i][j] * U[j][k]);
            U[i][k] = A[i][k] - sum;
        }

        for (int k = i + 1; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[k][j] * U[j][i]);
            L[k][i] = (A[k][i] - sum) / U[i][i];
        }
    }
}

void lu_parallel(int n, const vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U) {
    for (int i = 0; i < n; i++) {
#pragma omp parallel for schedule(dynamic) 
        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[i][j] * U[j][k]);
            U[i][k] = A[i][k] - sum;
        }

#pragma omp parallel for schedule(dynamic)
        for (int k = i + 1; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[k][j] * U[j][i]);
            L[k][i] = (A[k][i] - sum) / U[i][i];
        }
    }
}

int main() {
    vector<int> test_sizes = { 100, 200, 500, 1000, 1500 };

    cout << left << setw(10) << "Size(N)"
        << setw(15) << "Serial(s)"
        << setw(15) << "OpenMP(s)"
        << setw(15) << "Speedup"
        << setw(15) << "Efficiency(%)" << endl;
    cout << string(70, '-') << endl;

    int max_threads = 20;
    omp_set_num_threads(max_threads);
    srand(42);

    for (int n : test_sizes) {
        vector<vector<double>> A(n, vector<double>(n));
        vector<vector<double>> L(n, vector<double>(n));
        vector<vector<double>> U(n, vector<double>(n));

        initMatrix(n, A);

        resetLU(n, L, U);
        double start_s = omp_get_wtime();
        lu_serial(n, A, L, U);
        double end_s = omp_get_wtime();
        double time_s = end_s - start_s;

        resetLU(n, L, U);
        double start_p = omp_get_wtime();
        lu_parallel(n, A, L, U);
        double end_p = omp_get_wtime();
        double time_p = end_p - start_p;

        double speedup = time_s / time_p;
        double efficiency = (speedup / max_threads) * 100;

        cout << left << setw(10) << n
            << setw(15) << fixed << setprecision(5) << time_s
            << setw(15) << time_p
            << setw(15) << setprecision(2) << speedup
            << setw(15) << efficiency << endl;
    }

    cout << string(70, '-') << endl;
    cout << "Threads used: " << max_threads << endl;

    return 0;
}