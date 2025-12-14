#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <cstdlib>
#include <string>

using namespace std;

void initMatrix(int n, vector<vector<double>>& A, vector<double>& b) {
    for (int i = 0; i < n; i++) {
        double row_sum = 0;
        for (int j = 0; j < n; j++) {
            A[i][j] = (rand() % 10) + 1;
            row_sum += abs(A[i][j]);
        }
        A[i][i] += row_sum;

        b[i] = (rand() % 20) + 1;
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
#pragma omp parallel for schedule(guided) shared(A, L, U)
        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[i][j] * U[j][k]);
            U[i][k] = A[i][k] - sum;
        }

#pragma omp parallel for schedule(guided) shared(A, L, U)
        for (int k = i + 1; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[k][j] * U[j][i]);
            L[k][i] = (A[k][i] - sum) / U[i][i];
        }
    }
}

vector<double> solveSystem(int n, const vector<vector<double>>& L, const vector<vector<double>>& U, const vector<double>& b) {
    vector<double> y(n);
    vector<double> x(n);

    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) sum += L[i][j] * y[j];
        y[i] = (b[i] - sum);
    }

    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++) sum += U[i][j] * x[j];
        x[i] = (y[i] - sum) / U[i][i];
    }
    return x;
}

double validate(int n, const vector<vector<double>>& A, const vector<double>& x, const vector<double>& b) {
    double max_error = 0.0;
#pragma omp parallel for reduction(max:max_error)
    for (int i = 0; i < n; i++) {
        double ax_i = 0;
        for (int j = 0; j < n; j++) ax_i += A[i][j] * x[j];
        if (fabs(b[i] - ax_i) > max_error) max_error = fabs(b[i] - ax_i);
    }
    return max_error;
}

int main() {
    vector<int> test_sizes = { 100, 200, 500, 1000, 1500, 2000 };

    cout << left << setw(10) << "Size(N)"
        << setw(15) << "Serial(s)"
        << setw(15) << "OpenMP(s)"
        << setw(15) << "Speedup"
        << setw(15) << "Efficiency(%)"
        << setw(15) << "Status" << endl; 
    cout << string(85, '-') << endl;

    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    srand(42);

    for (int n : test_sizes) {
        vector<vector<double>> A(n, vector<double>(n));
        vector<double> b(n);

        vector<vector<double>> L_s(n, vector<double>(n));
        vector<vector<double>> U_s(n, vector<double>(n));
        vector<vector<double>> L_p(n, vector<double>(n));
        vector<vector<double>> U_p(n, vector<double>(n));

        initMatrix(n, A, b);

        resetLU(n, L_s, U_s);
        double start_s = omp_get_wtime();
        lu_serial(n, A, L_s, U_s);
        double end_s = omp_get_wtime();
        double time_s = end_s - start_s;

        resetLU(n, L_p, U_p);
        double start_p = omp_get_wtime();
        lu_parallel(n, A, L_p, U_p);
        double end_p = omp_get_wtime();
        double time_p = end_p - start_p;

        vector<double> x = solveSystem(n, L_p, U_p, b);
        double error = validate(n, A, x, b);

        string status = (error < 1e-9) ? "Correct" : "Not Correct";

        double speedup = time_s / time_p;
        double efficiency = (speedup / max_threads) * 100;

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