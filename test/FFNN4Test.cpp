#include "FFNN4Test.hpp"

void FFNN4Test::setUp()
{
    std::vector<double> input = {0.421155, -0.5615047, 1.462359};
    octaveInput = input;

    std::vector<std::vector<double>> w1 = {
        { 0.04881425,   -0.305647,  0.07962375},
        { 0.9021644,  -0.4051043,   -1.189009},
        { -0.6151073,  -0.01014279,   0.4181548},
        { 0.1344998,  0.03050448,    2.137649},
        { 0.1881913,   -1.090388,  -0.3023237},
        {-1.400147,  -0.003861771,    1.657677},
        {-2.010174,  -0.5720709,   -1.432056},
        {  1.031725,   0.9609774,    -1.11182},
        {-1.150198,   -1.764942,  -0.3193126},
        {-0.9868135,    0.571306,   -1.876868}
    };
    octaveMatrix1 = w1;

    std::vector<std::vector<double>> w2 = {
        {0.7900925,   0.6508365,  -0.5207233,  -0.0725704,  -0.06202245,   0.3435975,  -0.8378694,   0.8844505,   0.4922904, -0.3262369},
        { 2.476513,  -0.8987411,   0.6574533,  -0.4750641,   0.4344365,    1.340195,   -1.150354,   0.2278437,  -0.2577546, 1.302318},
        {-0.3786916,    1.226907,   0.4178262,   -1.333403,   -1.943758,    1.288567,  -0.1153724,   -1.399761,  -0.5080219, 0.279853},
        { 0.01416269,  -0.5529898,  -0.9178863,   0.7862638,  -0.7366727,    1.582841,  -0.5184778,  -0.8651988,   -2.231475, 0.6298529},
        {0.03054983,   0.2314384,  -0.8808459,    1.480631,  -0.1376327,  0.03127562,  -0.8770813,     0.61245,   0.7476056, 1.803704}
    };
    octaveMatrix2 = w2;

    std::vector<std::vector<double>> w3 = {
        { 0.1082638,   0.1623608,    1.099994,   0.2529691,  -0.4539777 },
        { 1.596734,  -0.5084351,  -0.7910981,   0.3171132,    1.672245},
        {-0.2924226,  -0.8344571,  -0.5897098,   0.5564259,  -0.5439654},
        {-2.156333,   -1.016486,   -1.265088,   0.4559662,   0.7881437},
        {-2.135809,   0.7033691,   0.5285266,  -0.8641451,  0.03606475},
        {-0.5679011,  -0.3760505,    1.220384,  -0.03969354,    1.862471},
        { 1.467235,   0.6441833,   0.3592659,  0.07126924,    1.839619}
    };
    octaveMatrix3 = w3;

    // Pre-calculated reference output.
    std::vector<double> refOutput1 = {0.5477323, 0.8882225, 0.2255278, 0.1389464, 0.2301051, 0.7339851, 0.9622299};
    octaveSigmoidOutput = refOutput1;

    std::vector<double> refOutput2 = {-0.1985417, 0.6819566, 0.1214852, -0.6027433, -0.9025843, -0.9269031, 0.6649065};
    octaveTanhOutput = refOutput2;

    std::vector<double> refOutput3 = {-0.166299, 0.7138456, 0.2530134, -0.6676853, -1.062403, -1.09945, 0.6430363};
    octaveArctanOutput = refOutput3;

    std::vector<double> refOutput4 = {1, 1, 0, 0, 0, 1, 1};
    octaveBinaryOutput = refOutput4;
}

void FFNN4Test::tearDown() {;}

void FFNN4Test::testConstructor()
{
    FFNN4 n1(-1, 1, 1, 1, SIGMOID);
}

void FFNN4Test::testBadInput()
{
    FFNN4 n1(3, 3, 3, 3, SIGMOID);
    std::vector<double> input = {1.121, 9.122};
    std::vector<double> result = n1.calculateOutput(input);
    CPPUNIT_ASSERT(1 == result.size());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result[0], 0.001);
}

void FFNN4Test::testOutputIdentity()
{
    FFNN4 n1(5, 3, 3, 2, IDENTITY);
    
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

void FFNN4Test::testOutputSigmoid()
{
    FFNN4 n1(3, 10, 5, 7, SIGMOID);

    n1.setWeightMatrices(octaveMatrix1, octaveMatrix2, octaveMatrix3);
    std::vector<double> netOut1 = n1.calculateOutput(octaveInput);
    CPPUNIT_ASSERT(octaveSigmoidOutput.size() == netOut1.size());
    for(unsigned int i = 0; i < netOut1.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(octaveSigmoidOutput[i], netOut1[i], 0.000001);
    }
}
void FFNN4Test::testOutputTanh()
{
    FFNN4 n1(3, 10, 5, 7, TANH);

    n1.setWeightMatrices(octaveMatrix1, octaveMatrix2, octaveMatrix3);
    std::vector<double> netOut1 = n1.calculateOutput(octaveInput);
    CPPUNIT_ASSERT(octaveTanhOutput.size() == netOut1.size());
    for(unsigned int i = 0; i < netOut1.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(octaveTanhOutput[i], netOut1[i], 0.000001);
    }
}
void FFNN4Test::testOutputArctan()
{
    FFNN4 n1(3, 10, 5, 7, ARCTAN);

    n1.setWeightMatrices(octaveMatrix1, octaveMatrix2, octaveMatrix3);
    std::vector<double> netOut1 = n1.calculateOutput(octaveInput);
    CPPUNIT_ASSERT(octaveArctanOutput.size() == netOut1.size());
    for(unsigned int i = 0; i < netOut1.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(octaveArctanOutput[i], netOut1[i], 0.000001);
    }
}

void FFNN4Test::testOutputBinary()
{
    FFNN4 n1(3, 10, 5, 7, BINARY);

    n1.setWeightMatrices(octaveMatrix1, octaveMatrix2, octaveMatrix3);
    std::vector<double> netOut1 = n1.calculateOutput(octaveInput);
    CPPUNIT_ASSERT(octaveBinaryOutput.size() == netOut1.size());
    for(unsigned int i = 0; i < netOut1.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(octaveBinaryOutput[i], netOut1[i], 0.000001);
    }
}
