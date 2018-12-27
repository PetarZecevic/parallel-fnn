#ifndef FFNN4_HPP
#define FFNN4_HPP

#include <vector>
#include "Layer.hpp"

/*
Class that models 4-layer Feed-Forward neural network.
*/
class FFNN4
{
    Layer inputLayer;
    Layer hiddenLayer1;
    Layer hiddenLayer2;
    Layer outputLayer;
    std::vector<float> input;
    std::vector<float> output;
    ActivationFunctionType activation;
public:
    FFNN4(unsigned int neuronsInput,unsigned int neuronsHidden1,
    	  unsigned int neuronsHidden2,unsigned int neuronsOutput, ActivationFunctionType act = SIGMOID);
    
    void setInput(const std::vector<float>& in);
    
    void setWeightMatrices(const std::vector< std::vector <float> >& w1, const std::vector< std::vector<float> >& w2, const std::vector< std::vector<float> >& w3);
    
    // Get inputs from input field and store result in output field.
    void calculateOutput(void);
    
    // Get inputs from in parameter and return results.
    std::vector<float> calculateOutput(const std::vector<float>& in);
    
    std::vector<float> getOutput(void) const;  
};

#endif // FFNN4_HPP
