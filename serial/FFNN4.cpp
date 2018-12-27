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
    
void FFNN4::setInput(const std::vector<float>& in)
{
    input = in;
}
void FFNN4::setWeightMatrices(const std::vector< std::vector <float> >& w1, const std::vector< std::vector<float> >& w2, const std::vector< std::vector<float> >& w3)
{
    hiddenLayer1.setWeightMatrix(w1);
    hiddenLayer2.setWeightMatrix(w2);
    outputLayer.setWeightMatrix(w3);
}
void FFNN4::calculateOutput(void)
{
    inputLayer.setInput(input);
    inputLayer.calculateOutput();
    std::vector<float> inputOut = inputLayer.getOutput();

    hiddenLayer1.setInput(inputOut);
    hiddenLayer1.calculateOutput();
    std::vector<float> hiddenOut1 = hiddenLayer1.getOutput();

    hiddenLayer2.setInput(hiddenOut1);
    hiddenLayer2.calculateOutput();
    std::vector<float> hiddenOut2 = hiddenLayer2.getOutput();

    outputLayer.setInput(hiddenOut2);
    outputLayer.calculateOutput();
    output = outputLayer.getOutput();
}

std::vector<float> FFNN4::calculateOutput(const std::vector<float>& in)
{
	inputLayer.setInput(in);
    inputLayer.calculateOutput();
    std::vector<float> inputOut = inputLayer.getOutput();

    hiddenLayer1.setInput(inputOut);
    hiddenLayer1.calculateOutput();
    std::vector<float> hiddenOut1 = hiddenLayer1.getOutput();

    hiddenLayer2.setInput(hiddenOut1);
    hiddenLayer2.calculateOutput();
    std::vector<float> hiddenOut2 = hiddenLayer2.getOutput();

    outputLayer.setInput(hiddenOut2);
    outputLayer.calculateOutput();
    return outputLayer.getOutput();
}

std::vector<float> FFNN4::getOutput(void) const
{
    return output;
}
