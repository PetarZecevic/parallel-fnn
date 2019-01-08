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
	
    LayerP hiddenLayer(100, 100, HIDDEN, SIGMOID);
    
    vector<vector <double> > w(100, vector<double>(100));
    RandInitMatrix(w);
    
    vector<double> input(100);
    RandInitVector(input);

    hiddenLayer.setWeightMatrix(w);
    hiddenLayer.setInput(input);
	
	
	tick_count startTime = tick_count::now();
    hiddenLayer.calculateOutput();
	tick_count endTime = tick_count::now();
	
	cout << "Parallel execution time :" << (endTime-startTime).seconds()*1000 << "ms." << endl;
	
	
	startTime = tick_count::now();
    hiddenLayer.calculateOutputSerial();
	endTime = tick_count::now();
	
	cout << "Serial execution time :" << (endTime-startTime).seconds()*1000 << "ms." << endl;
	
	/*
	vector< vector<double> > m (100, vector<double>(100));
	RandInitMatrix(m);
	PrintMatrix(m);	
	*/
	
    return 0;
}
