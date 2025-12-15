#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <cstdlib>
#include <string>

using namespace std;

// Function to initialize a diagonally dominant matrix A and vector b
void initMatrix(int n, vector<vector<double>>& A, vector<double>& b) {
    for (int i = 0; i < n; i++) {
        double row_sum = 0;
        for (int j = 0; j < n; j++) {
            // Fill A with random values (1 to 10)
            A[i][j] = (rand() % 10) + 1;
            row_sum += abs(A[i][j]);
        }
        // Make the matrix strictly diagonally dominant to guarantee LU decomposition stability
        A[i][i] += row_sum;

        // Initialize b with random values (1 to 20)
        b[i] = (rand() % 20) + 1;
    }
}

// Function to reset L and U matrices before decomposition
void resetLU(int n, vector<vector<double>>& L, vector<vector<double>>& U) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = 0.0;
            U[i][j] = 0.0;
            // Set diagonal elements of L to 1
            if (i == j) L[i][i] = 1.0;
        }
    }
}

// Serial LU Decomposition (Doolittle algorithm)
void lu_serial(int n, const vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U) {
    for (int i = 0; i < n; i++) {
        // Calculate the i-th row of U (U[i][k] for k >= i)
        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[i][j] * U[j][k]);
            U[i][k] = A[i][k] - sum;
        }

        // Calculate the i-th column of L (L[k][i] for k > i)
        // L[i][i] is implicitly 1
        for (int k = i + 1; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[k][j] * U[j][i]);
            L[k][i] = (A[k][i] - sum) / U[i][i];
        }
    }
}

// Parallel LU Decomposition using OpenMP
void lu_parallel(int n, const vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U) {
    for (int i = 0; i < n; i++) {

        // Parallel calculation of the i-th row of U
        // The inner loop (k) is independent across different values of k for a fixed i.
        // `schedule(guided)` is often good for loops with varying workload per iteration.
#pragma omp parallel for schedule(guided) shared(A, L, U)
        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[i][j] * U[j][k]);
            U[i][k] = A[i][k] - sum;
        }

        // Parallel calculation of the i-th column of L
        // The inner loop (k) is independent across different values of k for a fixed i.
#pragma omp parallel for schedule(guided) shared(A, L, U)
        for (int k = i + 1; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[k][j] * U[j][i]);
            L[k][i] = (A[k][i] - sum) / U[i][i];
        }
    }
}

// Solve Ax = b using the decomposed form LUx = b
// 1. Solve Ly = b (Forward Substitution)
// 2. Solve Ux = y (Backward Substitution)
vector<double> solveSystem(int n, const vector<vector<double>>& L, const vector<vector<double>>& U, const vector<double>& b) {
    vector<double> y(n);
    vector<double> x(n);

    // Step 1: Forward Substitution (Ly = b)
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) sum += L[i][j] * y[j];
        // L[i][i] is 1, so division is implicit: y[i] = (b[i] - sum) / 1
        y[i] = (b[i] - sum);
    }

    // Step 2: Backward Substitution (Ux = y)
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++) sum += U[i][j] * x[j];
        x[i] = (y[i] - sum) / U[i][i];
    }
    return x;
}

// Validate the solution x by calculating the maximum residual ||Ax - b||_inf
double validate(int n, const vector<vector<double>>& A, const vector<double>& x, const vector<double>& b) {
    double max_error = 0.0;
    // Parallelize the residual calculation
#pragma omp parallel for reduction(max:max_error)
    for (int i = 0; i < n; i++) {
        double ax_i = 0;
        for (int j = 0; j < n; j++) ax_i += A[i][j] * x[j];
        double current_error = fabs(b[i] - ax_i);
        if (current_error > max_error) max_error = current_error;
    }
    return max_error;
}

int main() {
    // Define matrix sizes for testing
    vector<int> test_sizes = { 100, 200, 500, 1000, 1500, 2000 };

    // Print header for results table
    cout << left << setw(10) << "Size(N)"
        << setw(15) << "Serial(s)"
        << setw(15) << "OpenMP(s)"
        << setw(15) << "Speedup"
        << setw(15) << "Efficiency(%)"
        << setw(15) << "Status" << endl;
    cout << string(85, '-') << endl;

    // Get maximum available threads and set the thread count
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    srand(42); // Seed for reproducibility

    for (int n : test_sizes) {
        // Allocate matrices and vectors
        vector<vector<double>> A(n, vector<double>(n));
        vector<double> b(n);

        vector<vector<double>> L_s(n, vector<double>(n));
        vector<vector<double>> U_s(n, vector<double>(n));
        vector<vector<double>> L_p(n, vector<double>(n));
        vector<vector<double>> U_p(n, vector<double>(n));

        // Initialize the input system
        initMatrix(n, A, b);

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

        // --- Validation (using parallel result) ---
        // Note: The solution phase (solveSystem) is not parallelized here
        // as its structure (forward/backward substitution) has high dependencies.
        vector<double> x = solveSystem(n, L_p, U_p, b);
        double error = validate(n, A, x, b);

        // Determine if the solution is accurate
        string status = (error < 1e-9) ? "Correct" : "Not Correct";

        // Calculate performance metrics
        double speedup = time_s / time_p;
        double efficiency = (speedup / max_threads) * 100;

        // Print results row
        cout << left << setw(10) << n
            << setw(15) << fixed << setprecision(5) << time_s
            << setw(15) << time_p
            << setw(15) << setprecision(2) << speedup
            << setw(15) << efficiency
            << setw(15) << status << endl;
    }

    cout << string(85, '-') << endl;
    cout << "Threads used: " << max_threads << endl;

    return 0;
}