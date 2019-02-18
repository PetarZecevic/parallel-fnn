#ifndef FFNN4PTEST_HPP
#define FFNN4PTEST_HPP

#include <cppunit/extensions/HelperMacros.h>
#include "FFNN4P.hpp"

class FFNN4PTest : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(FFNN4PTest);
    CPPUNIT_TEST_EXCEPTION(testConstructor, std::bad_alloc);
    CPPUNIT_TEST(testBadInput);
    CPPUNIT_TEST(testOutputIdentity);
    CPPUNIT_TEST(testOutputSigmoid);
    CPPUNIT_TEST(testOutputTanh);
    CPPUNIT_TEST(testOutputArctan);
    CPPUNIT_TEST(testOutputBinary);
    CPPUNIT_TEST_SUITE_END();
public:
    void setUp();
    void tearDown();
protected:
    void testConstructor();
    void testBadInput();
    void testOutputIdentity();
    void testOutputSigmoid();
    void testOutputTanh();
    void testOutputArctan();
    void testOutputBinary();
};

#endif
