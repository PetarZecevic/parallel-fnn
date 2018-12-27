#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <random>

#include "FFNN4.hpp"

using namespace std;

// Random number generator, probability distribution is normal distribution.
normal_distribution<float> urd1{-1.0, 1.0};
default_random_engine e;

void RandInitWeigthMatrix(vector < vector <float> >& w)
{
    for(size_t i = 0 ; i < w.size(); ++i)
    {
        for(size_t j = 0; j < w[i].size(); ++j)
        {
            w[i][j] = urd1(e);
        }
    }
}

void PrintMatrix(const vector < vector <float> >& w)
{
    for(size_t i = 0 ; i < w.size(); ++i)
    {
        for(size_t j = 0; j < w[i].size(); ++j)
        {
            cout << "( " << w[i][j] << " )" << " ";
        }
        cout << endl;
    }
}


int main(void)
{
    // Nerons per layer.
    vector<unsigned int> neurons = {10, 10, 10, 10};
    
    // Weight matrices.
    vector<vector<float> > w1(neurons[1], vector<float>(neurons[0]));
    vector< vector<float> > w2(neurons[2], vector<float>(neurons[1]));
    vector< vector<float> > w3(neurons[3], vector<float>(neurons[2]));
    
    // Initialize Matrices.
    RandInitWeigthMatrix(w1);
    RandInitWeigthMatrix(w2);
    RandInitWeigthMatrix(w3);
    
    // Random Network input.
    vector<float> input(neurons[0]);
    for(size_t i = 0; i < neurons[0]; i++)
    {
        input[i] = urd1(e);
    }
    
   	
    // Create Network.
    FFNN4 net(neurons[0], neurons[1], neurons[2], neurons[3]);
    
    net.setWeightMatrices(w1, w2, w3);
    vector<float> output = net.calculateOutput(input);

    // Print output.
    for(size_t i = 0; i < neurons[3]; i++)
    {
        cout << output[i] << endl;
    }
    cout << endl << endl;
    
    return 0;
}
