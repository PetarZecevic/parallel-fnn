#ifndef RANDOMIZE_HPP
#define RANDOMIZE_HPP

#include <cstdlib>
#include <vector>
#include <iostream>

// Scales pseudo-random number into [-1,1] range.
#define	RANDOM	(-1.0 +	2.0*(double)rand() / RAND_MAX)

/*
Initial assumption is that matrix is regular,
that means that all columns have same number of elements.
*/
void RandInitMatrix(std::vector< std::vector<double> >& matrix);
void RandInitVector(std::vector<double>& vec);

/*
Initial assumption is that matrix is regular,
that means that all columns have same number of elements.
*/
void PrintMatrix(const std::vector< std::vector<double> >& matrix);
void PrintVector(const std::vector<double>& vec);

#endif
