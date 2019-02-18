#include "randomize.hpp"

void RandInitMatrix(std::vector< std::vector<double> >& matrix)
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

void RandInitVector(std::vector<double>& vec)
{
    const unsigned int vecSize = vec.size();
    for(unsigned int i = 0; i < vecSize; i++)
    {
        vec[i] = RANDOM;
    }
}

void PrintMatrix(const std::vector< std::vector<double> >& matrix)
{
	const unsigned int rowNum = matrix.size();
	const unsigned int colNum = matrix[0].size();
	
	for(unsigned int i = 0; i < rowNum; i++)
    {
    	std::cout << "[ ";
        for(unsigned int j = 0; j < colNum; j++)
        {
            std::cout << "( " << matrix[i][j] << " ) ";
        }
        std::cout << "]" << std::endl;
    }
}

void PrintVector(const std::vector<double>& vec)
{
	const unsigned int vecSize = vec.size();
	std::cout << "[ ";
	for(unsigned int i = 0; i < vecSize; i++)
	{
		std::cout << "( " << vec[i] << " ) ";
	}
	std::cout << "]" << std::endl;
}

