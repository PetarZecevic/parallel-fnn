#include "FFNN4.hpp"

FFNN4::FFNN4(unsigned int neuronsInput,unsigned int neuronsHidden1,
			unsigned int neuronsHidden2,unsigned int neuronsOutput, ActivationFunctionType act):
    inputLayer(neuronsInput, neuronsInput, INPUT, act),
    hiddenLayer1(neuronsInput, neuronsHidden1, HIDDEN, act),
    hiddenLayer2(neuronsHidden1, neuronsHidden2, HIDDEN, act),
    outputLayer(neuronsHidden2, neuronsOutput, OUTPUT, act),
    input(neuronsInput),
    output(neuronsOutput)
    {
    	activation = act;
    }
    
void FFNN4::setInput(const std::vector<double>& in)
{   
    if(in.size() != input.size())
        return;
    else
        input = in;
}

void FFNN4::setWeightMatrices(const std::vector< std::vector <double> >& w1, const std::vector< std::vector<double> >& w2, const std::vector< std::vector<double> >& w3)
{
    // Check dimensions
    if(w1.size() != hiddenLayer1.getNeuronsNumLayer() ||
        w1[0].size() != hiddenLayer1.getNeuronsNumPrevLayer() ||
        w2.size() != hiddenLayer2.getNeuronsNumLayer() ||
        w2[0].size() != hiddenLayer2.getNeuronsNumPrevLayer() ||
        w3.size() != outputLayer.getNeuronsNumLayer() ||
        w3[0].size() != outputLayer.getNeuronsNumPrevLayer())
    {
            return;
    }
    else
    {
        hiddenLayer1.setWeightMatrix(w1);
        hiddenLayer2.setWeightMatrix(w2);
        outputLayer.setWeightMatrix(w3);
    }
}

void FFNN4::calculateOutput(void)
{
    std::vector<double> result1 = inputLayer.calculateOutput(input);
    std::vector<double> result2 = hiddenLayer1.calculateOutput(result1);
    std::vector<double> result3 = hiddenLayer2.calculateOutput(result2);
    output = outputLayer.calculateOutput(result3);
}

std::vector<double> FFNN4::calculateOutput(const std::vector<double>& in)
{
    if(in.size() != inputLayer.getNeuronsNumLayer())
        return std::vector<double>{0};
    else
    {
        std::vector<double> result1 = inputLayer.calculateOutput(in);
        std::vector<double> result2 = hiddenLayer1.calculateOutput(result1);
        std::vector<double> result3 = hiddenLayer2.calculateOutput(result2);
        return outputLayer.calculateOutput(result3);
    }
}

std::vector<double> FFNN4::getOutput(void) const
{
    return output;
}
