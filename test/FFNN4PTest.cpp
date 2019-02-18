#include "FFNN4PTest.hpp"
#include "FFNN4.hpp"
#include "randomize.hpp"

void FFNN4PTest::setUp() {;}

void FFNN4PTest::tearDown() {;}

void FFNN4PTest::testConstructor()
{
    FFNN4P n1(-1, 1, 1, 1, SIGMOIDP);
}

void FFNN4PTest::testBadInput()
{
    FFNN4P n1(3, 3, 3, 3, SIGMOIDP);
    std::vector<double> input = {1.121, 9.122};
    std::vector<double> result = n1.calculateOutput(input);
    CPPUNIT_ASSERT(1 == result.size());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result[0], 0.001);
}

void FFNN4PTest::testOutputIdentity()
{
    FFNN4P n1(5, 3, 3, 2, IDENTITYP);
    
    std::vector<double> input = {1.00, 2.00, 3.00, 4.00, 5.00};
    std::vector<std::vector<double>> w1 = {
        {1.0, 1.0, 1.0, 1.0, 1.0}, 
        {1.0, 1.0, 1.0, 1.0, 1.0}, 
        {1.0, 1.0, 1.0, 1.0, 1.0}};
    
    std::vector<std::vector<double>> w2 = {
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0},
    };

    std::vector<std::vector<double>> w3 = {
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0},
    };
    
    std::vector<double> referenceOutput = {135, 135};

    n1.setWeightMatrices(w1, w2, w3);
    std::vector<double> nOutput = n1.calculateOutput(input);
    CPPUNIT_ASSERT(2 == nOutput.size());
    for(unsigned int i = 0; i < nOutput.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(referenceOutput[i], nOutput[i], 0.001);
    }

    n1.setInput(input);
    n1.calculateOutput();
    std::vector<double> nOutput1 = n1.getOutput();
    CPPUNIT_ASSERT(2 == nOutput1.size());
    for(unsigned int i = 0; i < nOutput1.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(referenceOutput[i], nOutput1[i], 0.001);
    }
}


void FFNN4PTest::testOutputSigmoid()
{
    FFNN4P n1P(100, 100, 100, 100, SIGMOIDP);
	FFNN4 n1(100, 100, 100, 100, SIGMOID);
	
	std::vector< std::vector<double> > wMatrix1(100, std::vector<double>(100));
	std::vector< std::vector<double> > wMatrix2(100, std::vector<double>(100));
	std::vector< std::vector<double> > wMatrix3(100, std::vector<double>(100));
	
	std::vector<double> input(100);
	
	RandInitMatrix(wMatrix1);
	RandInitMatrix(wMatrix2);
	RandInitMatrix(wMatrix3);
	RandInitVector(input);
	
    n1.setWeightMatrices(wMatrix1, wMatrix2, wMatrix3);
    n1P.setWeightMatrices(wMatrix1, wMatrix2, wMatrix3);
    
    std::vector<double> netOut1 = n1.calculateOutput(input);
    std::vector<double> netOut1P = n1P.calculateOutput(input);
    
    CPPUNIT_ASSERT(netOut1.size() == netOut1P.size());
    
    for(unsigned int i = 0; i < netOut1.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(netOut1[i], netOut1P[i], 0.000001);
    }
}


void FFNN4PTest::testOutputTanh()
{
    FFNN4P n1P(100, 500, 700, 100, TANHP);
	FFNN4 n1(100, 500, 700, 100, TANH);
	
	std::vector< std::vector<double> > wMatrix1(500, std::vector<double>(100));
	std::vector< std::vector<double> > wMatrix2(700, std::vector<double>(500));
	std::vector< std::vector<double> > wMatrix3(100, std::vector<double>(700));
	
	std::vector<double> input(100);
	
	RandInitMatrix(wMatrix1);
	RandInitMatrix(wMatrix2);
	RandInitMatrix(wMatrix3);
	RandInitVector(input);
	
    n1.setWeightMatrices(wMatrix1, wMatrix2, wMatrix3);
    n1P.setWeightMatrices(wMatrix1, wMatrix2, wMatrix3);
    
    std::vector<double> netOut1 = n1.calculateOutput(input);
    std::vector<double> netOut1P = n1P.calculateOutput(input);
    
    CPPUNIT_ASSERT(netOut1.size() == netOut1P.size());
    
    for(unsigned int i = 0; i < netOut1.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(netOut1[i], netOut1P[i], 0.000001);
    }
}

void FFNN4PTest::testOutputArctan()
{
    FFNN4P n1P(10, 100, 7, 2, ARCTANP);
	FFNN4 n1(10, 100, 7, 2, ARCTAN);
	
	std::vector< std::vector<double> > wMatrix1(100, std::vector<double>(10));
	std::vector< std::vector<double> > wMatrix2(7, std::vector<double>(100));
	std::vector< std::vector<double> > wMatrix3(2, std::vector<double>(7));
	
	std::vector<double> input(10);
	
	RandInitMatrix(wMatrix1);
	RandInitMatrix(wMatrix2);
	RandInitMatrix(wMatrix3);
	RandInitVector(input);
	
    n1.setWeightMatrices(wMatrix1, wMatrix2, wMatrix3);
    n1P.setWeightMatrices(wMatrix1, wMatrix2, wMatrix3);
    
    std::vector<double> netOut1 = n1.calculateOutput(input);
    std::vector<double> netOut1P = n1P.calculateOutput(input);
    
    CPPUNIT_ASSERT(netOut1.size() == netOut1P.size());
    
    for(unsigned int i = 0; i < netOut1.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(netOut1[i], netOut1P[i], 0.000001);
    }
}

void FFNN4PTest::testOutputBinary()
{
    FFNN4P n1P(1000, 200, 700, 10, BINARYP);
	FFNN4 n1(1000, 200, 700, 10, BINARY);
	
	std::vector< std::vector<double> > wMatrix1(200, std::vector<double>(1000));
	std::vector< std::vector<double> > wMatrix2(700, std::vector<double>(200));
	std::vector< std::vector<double> > wMatrix3(10, std::vector<double>(700));
	
	std::vector<double> input(1000);
	
	RandInitMatrix(wMatrix1);
	RandInitMatrix(wMatrix2);
	RandInitMatrix(wMatrix3);
	RandInitVector(input);
	
    n1.setWeightMatrices(wMatrix1, wMatrix2, wMatrix3);
    n1P.setWeightMatrices(wMatrix1, wMatrix2, wMatrix3);
    
    std::vector<double> netOut1 = n1.calculateOutput(input);
    std::vector<double> netOut1P = n1P.calculateOutput(input);
    
    CPPUNIT_ASSERT(netOut1.size() == netOut1P.size());
    
    for(unsigned int i = 0; i < netOut1.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(netOut1[i], netOut1P[i], 0.000001);
    }
}
