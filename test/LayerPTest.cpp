#include "LayerPTest.hpp"
#include "Layer.hpp"
#include "randomize.hpp"

#include <cmath>

void LayerPTest::setUp() {;}

void LayerPTest::tearDown() {;}

void LayerPTest::testConstructor()
{
    LayerP l1(-1, 0, INPUTP, SIGMOIDP);
}

void LayerPTest::testSetWeight()
{
    LayerP l2(3, 3, HIDDENP, SIGMOIDP);
    std::vector<std::vector<double>> w1 = {{1.0 , 1.0 , 1.0}, {1.0 ,1.0, 1.0}, {1.0, 1.0, 1.0}};
    l2.setWeightMatrix(w1);
    std::vector<std::vector<double>> wret1 = l2.getWeightMatrix();

    for(unsigned int i = 0; i < 3; i++)
    {
        for(unsigned int j = 0; j < 3; j++)
        {
            CPPUNIT_ASSERT_EQUAL(w1[i][j], wret1[i][j]);
        }
    }

    // Trol layer with bad dimension matrix.
    std::vector<std::vector<double>> w2 = {{4.0 , 5.5 , 1.23}, {1.23 ,1.444, 4545.0}};
    l2.setWeightMatrix(w2);
    std::vector<std::vector<double>> wret2 = l2.getWeightMatrix();

    for(unsigned int i = 0; i < 3; i++)
    {
        for(unsigned int j = 0; j < 3; j++)
        {
            CPPUNIT_ASSERT_EQUAL(wret1[i][j], wret2[i][j]);
        }
    }
}

void LayerPTest::testInputLayer()
{
    LayerP inLayer(10, 6, INPUTP, SIGMOIDP);

    // Troll layer with wrong dimension input.
    std::vector<double> in1 = {0.1, 0, 0.112, 0.3444, -1.212, 1.2121, 1.668, 98, -100, 0};

    // Test output1
    // Layer expects input with 6 elements, because Input layer has same dimension's for input and output vector.
    std::vector<double> out1 = inLayer.calculateOutput(in1);
    CPPUNIT_ASSERT(1 == out1.size());
    CPPUNIT_ASSERT(0 == out1[0]);

    // Test both outputFunctions, compare output to serial version
	Layer inSerial(1, 100000, INPUT, SIGMOID);
	LayerP inParallel(1, 100000, INPUTP, SIGMOIDP);
	
	// Testing variables.
	std::vector<double> input(100000);
	
	// Randomize vector.
	RandInitVector(input);
	
	std::vector<double> outSerial = inSerial.calculateOutput(input);
	std::vector<double> outParallel = inParallel.calculateOutput(input);
	
	CPPUNIT_ASSERT(outSerial.size() == outParallel.size());
	for(unsigned int i = 0; i < outSerial.size(); i++)
	{
		CPPUNIT_ASSERT_DOUBLES_EQUAL(outSerial[i], outParallel[i], 0.000001);
	}

    inSerial.setInput(input);
    inParallel.setInput(input);

    inSerial.calculateOutput();
    inParallel.calculateOutput();

    std::vector<double> outSerial1 = inSerial.getOutput();
	std::vector<double> outParallel1 = inParallel.getOutput();
	
	CPPUNIT_ASSERT(outSerial1.size() == outParallel1.size());
	for(unsigned int i = 0; i < outSerial1.size(); i++)
	{
		CPPUNIT_ASSERT_DOUBLES_EQUAL(outSerial1[i], outParallel1[i], 0.000001);
	}
}

void LayerPTest::testHiddenLayer()
{
    Layer hSerial(10000, 100, HIDDEN, ARCTAN);
    LayerP hParallel(10000, 100, HIDDENP, ARCTANP);

    std::vector< std::vector<double> > w(100, std::vector<double>(10000));
    std::vector<double> in(10000);
    
    RandInitMatrix(w);
    RandInitVector(in);

    hSerial.setWeightMatrix(w);
    hParallel.setWeightMatrix(w);

    std::vector<double> outSerial = hSerial.calculateOutput(in);
	std::vector<double> outParallel = hParallel.calculateOutput(in);
	
	CPPUNIT_ASSERT(outSerial.size() == outParallel.size());
	for(unsigned int i = 0; i < outSerial.size(); i++)
	{
		CPPUNIT_ASSERT_DOUBLES_EQUAL(outSerial[i], outParallel[i], 0.000001);
	}

    hSerial.setInput(in);
    hParallel.setInput(in);

    hSerial.calculateOutput();
    hParallel.calculateOutput();

    std::vector<double> outSerial1 = hSerial.getOutput();
	std::vector<double> outParallel1 = hParallel.getOutput();
	
	CPPUNIT_ASSERT(outSerial1.size() == outParallel1.size());
	for(unsigned int i = 0; i < outSerial1.size(); i++)
	{
		CPPUNIT_ASSERT_DOUBLES_EQUAL(outSerial1[i], outParallel1[i], 0.000001);
	}
}

void LayerPTest::testOutputLayer()
{
    Layer oSerial(10000, 1, OUTPUT, TANH);
    LayerP oParallel(10000, 1, OUTPUTP, TANHP);

    std::vector< std::vector<double> > w(1, std::vector<double>(10000));
    std::vector<double> in(10000);
    
    RandInitMatrix(w);
    RandInitVector(in);

    oSerial.setWeightMatrix(w);
    oParallel.setWeightMatrix(w);

    std::vector<double> outSerial = oSerial.calculateOutput(in);
	std::vector<double> outParallel = oParallel.calculateOutput(in);
	
	CPPUNIT_ASSERT(outSerial.size() == outParallel.size());
	for(unsigned int i = 0; i < outSerial.size(); i++)
	{
		CPPUNIT_ASSERT_DOUBLES_EQUAL(outSerial[i], outParallel[i], 0.000001);
	}

    oSerial.setInput(in);
    oParallel.setInput(in);

    oSerial.calculateOutput();
    oParallel.calculateOutput();

    std::vector<double> outSerial1 = oSerial.getOutput();
	std::vector<double> outParallel1 = oParallel.getOutput();
	
	CPPUNIT_ASSERT(outSerial1.size() == outParallel1.size());
	for(unsigned int i = 0; i < outSerial1.size(); i++)
	{
		CPPUNIT_ASSERT_DOUBLES_EQUAL(outSerial1[i], outParallel1[i], 0.000001);
	}
}
