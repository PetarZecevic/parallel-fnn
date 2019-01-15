#include <iostream>
#include <cstdlib>
#include "FFNN4P.hpp"
#include "tbb/tick_count.h"

// Scales pseudo-random number into [-1,1] range.
#define	RANDOM	(-1.0 +	2.0*(double)rand() / RAND_MAX)

using namespace std;
using namespace tbb;

/*
Initial assumption is that matrix is regular,
that means that all columns has same number of elements.
*/
void RandInitMatrix(vector< vector<double> >& matrix)
{
    const unsigned int rowNum = matrix.size();
    const unsigned int colNum = matrix[0].size();
    for(unsigned int i = 0; i < rowNum; i++)
    {
        for(unsigned int j = 0; j < colNum; j++)
        {
            matrix[i][j] = RANDOM;
        }
    }
}

void RandInitVector(vector<double>& vec)
{
    const unsigned int vecSize = vec.size();
    for(unsigned int i = 0; i < vecSize; i++)
    {
        vec[i] = RANDOM;
    }
}

void PrintMatrix(const vector< vector<double> >& matrix)
{
	const unsigned int rowNum = matrix.size();
	const unsigned int colNum = matrix[0].size();
	
	for(unsigned int i = 0; i < rowNum; i++)
    {
    	cout << "[ ";
        for(unsigned int j = 0; j < colNum; j++)
        {
            cout << "( " << matrix[i][j] << " ) ";
        }
        cout << "]" << endl;
    }
}

int main(void)
{   
    srand(time(NULL));

    unsigned int n1 = 147;
    unsigned int n2 = 10123;
    unsigned int n3 = 763;
    unsigned int n4 = 41;

    testSubmatrices(n1, 1);
    testSubmatrices(n2, n1);
    testSubmatrices(n3, n2);
    testSubmatrices(n4, n3);

    vector<vector<double>> w1(n2, vector<double>(n1));
    vector<vector<double>> w2(n3, vector<double>(n2));
    vector<vector<double>> w3(n4, vector<double>(n3));
    
    vector<double> input(n1);

    RandInitMatrix(w1);
    RandInitMatrix(w2);
    RandInitMatrix(w3);
    RandInitVector(input);
       
    FFNN4P netP(n1, n2, n3, n4, SIGMOIDP);

    netP.setWeightMatrices(w1, w2, w3);

    tbb::tick_count start, end;
 
    start = tbb::tick_count::now();
    vector<double> output1 = netP.calculateOutput(input);
    end = tbb::tick_count::now();
    cout << "Time: " << (end - start).seconds() * 1000 << endl;

    return 0;
}
