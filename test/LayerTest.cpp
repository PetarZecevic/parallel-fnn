#include "LayerTest.hpp"

#include <cmath>

void LayerTest::setUp() 
{
    // Input-Matrix-Output-1
    std::vector<double> input1 = {-2.01, -3.55, 0.04, 0.05, -13.22, -2.8, -2.1, 0.0001, -0.11212, 1.00001};
    octaveInput1 = input1;
    std::vector< std::vector<double> > w1 = {
        {0.100, 0.200, 0.300, 0.100, 1.300, 4.100, -5.500, -2.100 ,1.140, 1.11111},
        {-0.110, -10.100, -2.300, -14.00, 12.00, 1.100, 4.800, 19.120, -0.00001, 0.111},
        {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
        {1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00},
        {-1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00},
        {0.0001, -0.0001, 1.010, -1.010, 34440.00, 20200.00, -1.002020, 1.0000001, 1.12000, 1.21212},
        {0.100, -0.110, -10.20, -2.30, -14.00, 12.00, 34440.00, -12020.00, 12.00, 5.00},  
    };

    octaveMatrix1.resize(7);
    for(unsigned int i = 0; i < 7; i++)
    {
        octaveMatrix1[i] = w1[i];
    }

    std::vector<double> output1 = { -17.02691, -136.403, 0, -22.70201, 22.70201, -511853.619, -72170.40089};
    octaveOutput1 = output1;

    // Input-Matrix-Output-2, Normal distribution.
    std::vector<double> input2 = {-0.0005851669, -1.05189, 1.366087,  1.432912, -0.1359012, 2.246143, -1.001619,
        0.1295635, 1.40751,  -0.4509072,   -0.146216,   0.6907739,   0.1369635,    0.029882,
         -0.9164035, 0.4604615,  -0.2719787,  0.361832,  0.2331528,  -1.025189};

    octaveInput2 = input2;

    std::vector<std::vector<double>> w2 = {
        {
         -0.9856119,  -1.171345,     1.84105,  -0.5505324,  -0.4074612,   -1.255239,    1.035557,  -0.8304458,   0.1746546,
          0.4982164,  0.01697169,  -0.3270149,   0.8582416,    1.002392,   -1.289917,   -1.591402,   0.4952099,    1.095131,
         -0.5322516,    1.057446
        },
        {
            -0.7882598,    1.231561,   -1.534408,  -0.2154968,  0.02158333,  -0.7862569,   -1.552038,   0.2134951,    1.098304,
            -0.7135044,  -0.8849015,    1.401876,    0.660495,   0.4295719,   0.6974267,  -0.9157153,  -0.4635791,   0.4594931,
            0.012306,   0.9730226
        },
        {
            -0.5804689,    1.970845,    1.229429,    1.201618,  -0.4659901,   0.2686976,   -1.036396,  0.07839593, -0.6960313,
             1.416646,   -1.172883,   0.1582607,  0.07936587,    1.494739,    1.876879,   -1.525899,   0.8055586,  -0.3259634,
             0.2213418,    -1.09552
        },
        {
             2.206478,   -1.238049,   0.2728906,   -1.490778,   -1.052146,  -0.06687429,   0.2446184,   0.1849905,   0.7628017,
             -0.1333155,  -0.1705179,   0.9258893,  0.06367175,   0.3613178,  -0.6877274,   0.1814552,     -1.9097,  -0.3623892,
             -0.3086188,   0.2382203
        },
        {
            -0.121681,    1.157294,   0.7408139,    2.505664,  -0.7201299,    2.131962,    1.245146,   0.7143021,    0.623713,
            -1.403665,   0.1537061,    1.257418,     1.29566,    1.174203,  -0.07721719,  -0.2134864,  -0.4755632,  -0.3427725,
             -0.3909229,  -0.05358567
        }
    };

    octaveMatrix2.resize(5);
    for(unsigned int i = 0; i < 5; i++)
    {
        octaveMatrix2[i] = w2[i];
    }

    std::vector<double> output2 = { -1.507028, -2.581126, 0.1766321, 1.913052, 9.628116};
    
    octaveOutput2 = output2;
}

void LayerTest::tearDown() {;}

void LayerTest::testConstructor()
{
    Layer l1(-1, 0, INPUT, SIGMOID);
}

void LayerTest::testSetWeight()
{
    Layer l2(3, 3, HIDDEN, SIGMOID);
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

void LayerTest::testInputLayer()
{
    Layer inLayer(10, 6, INPUT, SIGMOID);

    // Troll layer with wrong dimension input.
    std::vector<double> in1 = {0.1, 0, 0.112, 0.3444, -1.212, 1.2121, 1.668, 98, -100, 0};

    // Test output1
    // Layer expects input with 6 elements, because Input layer has same dimension's for input and output vector
    std::vector<double> out1 = inLayer.calculateOutput(in1);
    CPPUNIT_ASSERT(1 == out1.size());
    CPPUNIT_ASSERT(0 == out1[0]);

    // Test output2
    std::vector<double> in2 = {1.111, -1.111, 0.001, -0.001};
    Layer inLayer1(12129, 4, INPUT, SIGMOID);
    
    inLayer1.setInput(in2);
    
    // Test both outputFunctions
    std::vector<double> out2 = inLayer1.calculateOutput(in2);
    CPPUNIT_ASSERT(4 == out2.size());
    for(unsigned int i = 0; i < out2.size(); i++)
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1 / (1 + exp(-in2[i])), 1.0 * out2[i], 0.001);
    
    inLayer1.calculateOutput();
    std::vector<double> out22 = inLayer1.getOutput();
    CPPUNIT_ASSERT(4 == out22.size());
    for(unsigned int i = 0; i < out22.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1 / (1 + exp(-in2[i])), 1.0 * out22[i], 0.001);
    }
}

void LayerTest::testHiddenLayer()
{
    Layer hLayer( octaveInput1.size() , octaveOutput1.size() , HIDDEN, IDENTITY);
    hLayer.setWeightMatrix(octaveMatrix1);
    std::vector<double> layerOut1 = hLayer.calculateOutput(octaveInput1);

    CPPUNIT_ASSERT(layerOut1.size() == octaveOutput1.size());
    for(unsigned int i = 0; i < layerOut1.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(octaveOutput1[i], layerOut1[i], 0.001);
    }

    hLayer.setInput(octaveInput1);
    hLayer.calculateOutput();
    std::vector<double> layerOut2 = hLayer.getOutput();
    CPPUNIT_ASSERT(layerOut2.size() == octaveOutput1.size());
    for(unsigned int i = 0; i < layerOut2.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(octaveOutput1[i], layerOut2[i], 0.001);
    }
    
}

void LayerTest::testOutputLayer()
{
    Layer outLayer(octaveInput2.size(), octaveOutput2.size(), OUTPUT, IDENTITY);

    outLayer.setWeightMatrix(octaveMatrix2);
    std::vector<double> layerOut1 = outLayer.calculateOutput(octaveInput2);
    CPPUNIT_ASSERT(layerOut1.size() == octaveOutput2.size());
    for(unsigned int i = 0; i < layerOut1.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(octaveOutput2[i], layerOut1[i], 0.000001);
    }

    outLayer.setInput(octaveInput2);
    outLayer.calculateOutput();
    std::vector<double> layerOut2 = outLayer.getOutput();
    CPPUNIT_ASSERT(layerOut2.size() == octaveOutput2.size());
    for(unsigned int i = 0; i < layerOut2.size(); i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(octaveOutput2[i], layerOut2[i], 0.000001);
    }
}
