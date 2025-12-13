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
        // Calculate U[i][k] for k >= i (Row i of U)
        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[i][j] * U[j][k]);
            U[i][k] = A[i][k] - sum;
        }

        // Calculate L[k][i] for k > i (Column i of L)
        for (int k = i + 1; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[k][j] * U[j][i]);
            L[k][i] = (A[k][i] - sum) / U[i][i];
        }
    }
}

// Optimized Parallel LU Decomposition (Doolittle)
void lu_parallel(int n, const vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U) {
    for (int i = 0; i < n; i++) {

        // 1. Calculate U[i][k] for k >= i (Row i of U)
        // Parallelize the K loop. Use 'guided' scheduling for potentially better performance 
        // than 'dynamic' on large, decreasing loop sizes.
#pragma omp parallel for schedule(guided) private(k) shared(A, L, U)
        for (int k = i; k < n; k++) {
            double sum = 0;
            // The inner j loop (dot product) is sequential
            for (int j = 0; j < i; j++)
                sum += (L[i][j] * U[j][k]);
            U[i][k] = A[i][k] - sum;
        }

        // 2. Calculate L[k][i] for k > i (Column i of L)
        // Parallelize the K loop.
        // Note: U[i][i] must be non-zero (guaranteed by matrix initialization).
#pragma omp parallel for schedule(guided) private(k) shared(A, L, U)
        for (int k = i + 1; k < n; k++) {
            double sum = 0;
            // The inner j loop (dot product) is sequential
            for (int j = 0; j < i; j++)
                sum += (L[k][j] * U[j][i]);
            L[k][i] = (A[k][i] - sum) / U[i][i];
        }
    }
}

int main() {
    vector<int> test_sizes = { 100, 200, 500, 1000, 1500, 2000 };

    cout << left << setw(10) << "Size(N)"
        << setw(15) << "Serial(s)"
        << setw(15) << "OpenMP(s)"
        << setw(15) << "Speedup"
        << setw(15) << "Efficiency(%)" << endl;
    cout << string(70, '-') << endl;

    int max_threads = omp_get_max_threads();
    // Setting a fixed number of threads for consistency in benchmarking
    // If max_threads is low, we might cap it higher for better speedup demonstration, 
    // but using the system's max threads is standard practice. Let's ensure it's at least 4.
    if (max_threads < 4) max_threads = 4;
    omp_set_num_threads(max_threads);

    srand(42);

    for (int n : test_sizes) {
        vector<vector<double>> A(n, vector<double>(n));
        vector<vector<double>> L_s(n, vector<double>(n)); // For Serial
        vector<vector<double>> U_s(n, vector<double>(n));

        vector<vector<double>> L_p(n, vector<double>(n)); // For Parallel
        vector<vector<double>> U_p(n, vector<double>(n));


        initMatrix(n, A);

        // --- Serial Execution ---
        resetLU(n, L_s, U_s);
        double start_s = omp_get_wtime();
        lu_serial(n, A, L_s, U_s);
        double end_s = omp_get_wtime();
        double time_s = end_s - start_s;

        // --- Parallel Execution ---
        resetLU(n, L_p, U_p);
        double start_p = omp_get_wtime();
        lu_parallel(n, A, L_p, U_p);
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