#include "ActivationTest.hpp"
#include <cmath>

void ActivationTest::setUp()
{
    sigmoid = new Sigmoid();
    tanH = new TanH();
    arcTan = new ArcTan();
    binary = new BinaryStep();
    identity = new Identity();
}

void ActivationTest::tearDown()
{
    delete sigmoid;
    delete tanH;
    delete arcTan;
    delete binary;
    delete identity;
}

void ActivationTest::testSigmoid()
{
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.50000, sigmoid->calculate(0.0), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.49998, sigmoid->calculate(0.0001),  0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.75421, sigmoid->calculate(1.1212), 0.001);
}

void ActivationTest::testTanh()
{
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, tanH->calculate(0.0), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0001, tanH->calculate(0.0001),  0.0001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.807985, tanH->calculate(-1.1212), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.996117, tanH->calculate(3.1212), 0.001); 

}

void ActivationTest::testArctan()
{
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, arcTan->calculate(0.0), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.000099, arcTan->calculate(-0.0001),  0.000001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.8424735, arcTan->calculate(1.1212), 0.00001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.260740, arcTan->calculate(3.1212), 0.001); 

}

void ActivationTest::testBinary()
{
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, binary->calculate(0), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, binary->calculate(-0.0001),  0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, binary->calculate(0.0001),  0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, binary->calculate(5.0001),  0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, binary->calculate(-1.1212), 0.001);
}

void ActivationTest::testIdentity()
{
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.543 , identity->calculate(4.543), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.12187, identity->calculate(0.12187),  0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.121, identity->calculate(-1.121), 0.001);
}
