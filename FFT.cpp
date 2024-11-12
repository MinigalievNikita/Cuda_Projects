#include <iostream> 
#include <complex>
#include <vector> 
#include <cmath>
#include <time.h>
using namespace std;
const double PI = 3.141592653589793238460;


void fft(vector<complex<double>>& x)
{
    int n = x.size();
    // Base case 
    if (n <= 1) return;
    // Divide 
    vector<complex<double>> even(n / 2), odd(n / 2);
    for (int i = 0; i < n / 2; ++i)
    {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }
    // Conquer
    fft(even);
    fft(odd);
    // Combine 
    for (int k = 0; k < n / 2; ++k)
    {
        complex<double> t = polar(1.0, -2 * PI * k / n) * odd[k];
        x[k] = even[k] + t;
        x[k + n / 2] = even[k] - t;
    }
}



void printComplexVector(const vector<complex<double>>& vec)
{
    for (const auto& c : vec)
    {
        cout << c << endl;
    }
}




int main()
{
    // Must be a power of 2 
    const int N = 32768;
    double sumtime = 0;
    clock_t tStart;

    for (int j = 0; j < 10; ++j)
    {
        vector<complex<double>> data(N);
        for (int i = 0; i < N; ++i)
        {
            data[i] = i;
        }
        tStart = clock();

        fft(data);

        sumtime += (double)(clock() - tStart) / CLOCKS_PER_SEC;

        cout << "FFT output:" << endl;
        for (int i = 0; i < 8; ++i)
        {
            cout << data[i] << endl;
        }
    }

    printf("Time taken: %f s\n", sumtime/10);
    return 0;
}
