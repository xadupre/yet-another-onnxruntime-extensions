#include "yaourt_helpers.h"
#include <gtest/gtest.h>

using namespace yaourt_helpers;

TEST(TestMakeString, EmptyString) {
  std::string result = MakeString("");
  EXPECT_EQ(result, "");
}

TEST(TestMakeString, SingleString) {
  std::string result = MakeString("hello");
  EXPECT_EQ(result, "hello");
}

TEST(TestMakeString, ConcatStrings) {
  std::string result = MakeString("hello", " ", "world");
  EXPECT_EQ(result, "hello world");
}

TEST(TestMakeString, IntTypes) {
  EXPECT_EQ(MakeString(static_cast<int32_t>(42)), "42");
  EXPECT_EQ(MakeString(static_cast<int64_t>(-7)), "-7");
  EXPECT_EQ(MakeString(static_cast<int16_t>(3)), "3");
}

TEST(TestMakeString, UintTypes) {
  EXPECT_EQ(MakeString(static_cast<uint32_t>(10)), "10");
  EXPECT_EQ(MakeString(static_cast<uint64_t>(99)), "99");
  EXPECT_EQ(MakeString(static_cast<uint16_t>(5)), "5");
}

TEST(TestMakeString, FloatTypes) {
  std::string sf = MakeString(1.0f);
  EXPECT_FALSE(sf.empty());
  std::string sd = MakeString(2.5);
  EXPECT_FALSE(sd.empty());
}

TEST(TestMakeString, CharType) {
  std::string result = MakeString('A');
  EXPECT_EQ(result, "A");
}

TEST(TestMakeString, MixedTypes) {
  std::string result = MakeString("value=", static_cast<int32_t>(7));
  EXPECT_EQ(result, "value=7");
}

TEST(TestMakeString, NullptrUint64) {
  const uint64_t *p = nullptr;
  std::string result = MakeString(p);
  EXPECT_EQ(result, "(ui64*)null");
}

TEST(TestMakeString, VectorInt32) {
  std::vector<int32_t> v = {1, 2, 3};
  std::string result = MakeString(v);
  EXPECT_NE(result.find("1"), std::string::npos);
  EXPECT_NE(result.find("2"), std::string::npos);
  EXPECT_NE(result.find("3"), std::string::npos);
}

TEST(TestMakeString, VectorInt64) {
  std::vector<int64_t> v = {10, 20};
  std::string result = MakeString(v);
  EXPECT_NE(result.find("10"), std::string::npos);
  EXPECT_NE(result.find("20"), std::string::npos);
}

TEST(TestMakeString, VectorFloat) {
  std::vector<float> v = {1.0f, 2.0f};
  std::string result = MakeString(v);
  EXPECT_FALSE(result.empty());
}

TEST(TestMakeString, VectorDouble) {
  std::vector<double> v = {3.14};
  std::string result = MakeString(v);
  EXPECT_FALSE(result.empty());
}

TEST(TestMakeString, VectorUint16) {
  std::vector<uint16_t> v = {1, 2};
  std::string result = MakeString(v);
  EXPECT_FALSE(result.empty());
}

TEST(TestMakeString, VectorUint32) {
  std::vector<uint32_t> v = {100, 200};
  std::string result = MakeString(v);
  EXPECT_NE(result.find("100"), std::string::npos);
  EXPECT_NE(result.find("200"), std::string::npos);
}

TEST(TestMakeString, VectorUint64) {
  std::vector<uint64_t> v = {7, 8};
  std::string result = MakeString(v);
  EXPECT_FALSE(result.empty());
}

TEST(TestMakeString, VectorInt16) {
  std::vector<int16_t> v = {-1, 2};
  std::string result = MakeString(v);
  EXPECT_FALSE(result.empty());
}

TEST(TestSplitString, SplitByComma) {
  auto parts = SplitString("a,b,c", ',');
  ASSERT_EQ(parts.size(), 3u);
  EXPECT_EQ(parts[0], "a");
  EXPECT_EQ(parts[1], "b");
  EXPECT_EQ(parts[2], "c");
}

TEST(TestSplitString, NoDelimiter) {
  auto parts = SplitString("abc", ',');
  ASSERT_EQ(parts.size(), 1u);
  EXPECT_EQ(parts[0], "abc");
}

TEST(TestSplitString, EmptyString) {
  auto parts = SplitString("", ',');
  ASSERT_EQ(parts.size(), 1u);
  EXPECT_EQ(parts[0], "");
}

TEST(TestSplitString, TrailingDelimiter) {
  auto parts = SplitString("a,b,", ',');
  ASSERT_EQ(parts.size(), 3u);
  EXPECT_EQ(parts[0], "a");
  EXPECT_EQ(parts[1], "b");
  EXPECT_EQ(parts[2], "");
}

TEST(TestSplitString, LeadingDelimiter) {
  auto parts = SplitString(",a,b", ',');
  ASSERT_EQ(parts.size(), 3u);
  EXPECT_EQ(parts[0], "");
  EXPECT_EQ(parts[1], "a");
  EXPECT_EQ(parts[2], "b");
}

TEST(TestExtThrow, ThrowsRuntimeError) {
  EXPECT_THROW({ EXT_THROW("error message"); }, std::runtime_error);
}

TEST(TestExtThrow, MessageContainsYaourt) {
  try {
    EXT_THROW("something went wrong");
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("[yaourt]"), std::string::npos);
  }
}

TEST(TestExtThrow, MessageContainsUserText) {
  try {
    EXT_THROW("custom error text");
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("custom error text"), std::string::npos);
  }
}

TEST(TestExtEnforce, PassingConditionDoesNotThrow) {
  EXPECT_NO_THROW({ EXT_ENFORCE(1 == 1, "should not throw"); });
}

TEST(TestExtEnforce, FailingConditionThrowsRuntimeError) {
  EXPECT_THROW({ EXT_ENFORCE(1 == 2, "enforce failed"); }, std::runtime_error);
}

TEST(TestExtEnforce, ErrorMessageContainsConditionText) {
  try {
    EXT_ENFORCE(1 == 2, "my message");
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    std::string what = e.what();
    EXPECT_NE(what.find("1 == 2"), std::string::npos);
  }
}

TEST(TestExtEnforce, ErrorMessageContainsUserMessage) {
  try {
    EXT_ENFORCE(false, "user message here");
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("user message here"), std::string::npos);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
