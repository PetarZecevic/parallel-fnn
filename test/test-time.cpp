#include <iostream>
#include <cstdlib>
#include <fstream>
#include "tbb/tick_count.h"
#include "Layer.hpp"
#include "FFNN4.hpp"
#include "LayerP.hpp"
#include "FFNN4P.hpp"
#include "randomize.hpp"

using namespace std;
using namespace tbb;

void testFFNN4Time(unsigned int begin_size, unsigned int step_size, 
						   unsigned int maximum_size)
{
	fstream serialTimeFile;
	fstream parallelTimeFile;
    serialTimeFile.open("serial-time.txt", ios_base::out);
    parallelTimeFile.open("parallel-time.txt", ios_base::out);
	if(serialTimeFile.is_open() && parallelTimeFile.is_open())
	{
        tick_count startTime, endTime;
        serialTimeFile << begin_size << " " << step_size << " " << maximum_size << endl;
        parallelTimeFile << begin_size << " " << step_size << " " << maximum_size << endl;
        for(unsigned int current_size = begin_size; current_size <= maximum_size; current_size += step_size)
        {
            vector<vector <double> > w1(current_size, vector<double>(current_size));
            vector<vector <double> > w2(current_size, vector<double>(current_size));
            vector<vector <double> > w3(current_size, vector<double>(current_size));
            vector<double> input(current_size);

            RandInitMatrix(w1);
            RandInitMatrix(w2);
            RandInitMatrix(w3);
            RandInitVector(input);
            
            { // Serial block
                FFNN4 netSerial(current_size, current_size, current_size, current_size, SIGMOID);
                netSerial.setWeightMatrices(w1, w2, w3);
                netSerial.setInput(input);
                startTime = tick_count::now();
                netSerial.calculateOutput();
                endTime = tick_count::now();
                serialTimeFile << (endTime-startTime).seconds()*1000 << endl;
            }
            
            { // Parallel block
                FFNN4P netParallel(current_size, current_size, current_size, current_size, SIGMOIDP);
                netParallel.setWeightMatrices(w1, w2, w3);
                netParallel.setInput(input);

                startTime = tick_count::now();
                netParallel.calculateOutput();
                endTime = tick_count::now();
                parallelTimeFile << (endTime-startTime).seconds()*1000 << endl;
            }
        }
		serialTimeFile.close();
        parallelTimeFile.close();
	}	
}

int main(void)
{   
    testFFNN4Time(100, 500, 7100);
    return 0;
}
