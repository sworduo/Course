package funsets

import org.junit._

/**
 * This class is a test suite for the methods in object FunSets.
 *
 * To run this test suite, start "sbt" then run the "test" command.
 */
class FunSetSuite {

  import FunSets._

  @Test def `contains is implemented`: Unit = {
    assert(contains(x => true, 100))
  }

  /**
   * When writing tests, one would often like to re-use certain values for multiple
   * tests. For instance, we would like to create an Int-set and have multiple test
   * about it.
   *
   * Instead of copy-pasting the code for creating the set into every test, we can
   * store it in the test class using a val:
   *
   *   val s1 = singletonSet(1)
   *
   * However, what happens if the method "singletonSet" has a bug and crashes? Then
   * the test methods are not even executed, because creating an instance of the
   * test class fails!
   *
   * Therefore, we put the shared values into a separate trait (traits are like
   * abstract classes), and create an instance inside each test method.
   *
   */

  trait TestSets {
    val s1 = singletonSet(1)
    val s2 = singletonSet(2)
    val s3 = singletonSet(3)
    val s4 = singletonSet(4)
    val s5 = singletonSet(5)
    val s7 = singletonSet(7)
    val s1000 = singletonSet(1000)
    val u1 = union(s1, union(s2, s3))
    val u2 = union(s5, union(s2, s3))
  }

  /**
   * This test is currently disabled (by using @Ignore) because the method
   * "singletonSet" is not yet implemented and the test would fail.
   *
   * Once you finish your implementation of "singletonSet", remvoe the
   * @Ignore annotation.
   */
//  @Ignore("not ready yet") @Test def `singleton set one contains one`: Unit = {
  @Test def `singleton set one contains one`: Unit = {

    /**
     * We create a new instance of the "TestSets" trait, this gives us access
     * to the values "s1" to "s3".
     */
    new TestSets {
      /**
       * The string argument of "assert" is a message that is printed in case
       * the test fails. This helps identifying which assertion failed.
       */
      assert(contains(s1, 1), "Singleton")
    }
  }

  @Test def `union contains all elements of each set`: Unit = {
    new TestSets {
      val s = union(s1, s2)
      assert(contains(s, 1), "Union 1")
      assert(contains(s, 2), "Union 2")
      assert(!contains(s, 3), "Union 3")
    }

  }
  @Test def `union {1,3,4,5,7,1000} and {1,2,3,4}`: Unit ={
    new TestSets {
      val un1 = union(s1, union(s3, union(s4, union(s5, union(s7, s1000)))))
      val un2 = union(s1, union(s2, union(s3, s4)))
      val u3 = union(un1, un2)
      assert(contains(u3, 1000), "union 1000")
      assert(contains(u3, 7), "union 7")
    }
  }

  @Test def `diff test`: Unit={
    new TestSets {
      val d1 = diff(u1, u2)
      assert(contains(d1, 1), "diff 1")
      assert(!contains(d1, 2), "diff 2")
      assert(!contains(d1, 5), "diff 5")
    }
  }

  @Test def `interset test`: Unit={
    new TestSets {
      val i1 = intersect(u1, u2)
      assert(contains(i1, 2), "diff 2")
      assert(contains(i1, 3), "diff 3")
      assert(!contains(i1, 1), "diff 1")
      assert(!contains(i1, 5), "diff 5")
    }
  }

  @Test def `filter test`: Unit={
    new TestSets {
      val f1 = filter(u1, x=>x>1)
      assert(contains(f1, 2), "filter 2")
      assert(contains(f1, 3), "filter 3")
      assert(!contains(f1, 1), "filter 1")
    }
  }

  @Test def `forall test`: Unit={
    new TestSets {
      val f1 = union(u1, u2)
      assert(forall(f1, x=>x>0), "forall 1")
      assert(forall(f1, x=>x<10), "forall 2")
      assert(!forall(f1, x=>x>3), "forall 3")

    }
  }

  @Test def `exists test`: Unit={
    new TestSets {
      val f1 = union(u1, u2)
      assert(exists(f1, x=>x>0), "forall 1")
      assert(!exists(f1, x=>x<1), "forall 2")
      assert(exists(f1, x=>x>3), "forall 3")

    }
  }

  @Test def `map test`: Unit={
    new TestSets {
      val m1 = map(u1, x=> -x)
      assert(!contains(m1, 1), "map 1")
      assert(contains(m1, -1), "map 2")
      assert(contains(m1, -2), "map 3")
      val m2 = map(u1, x=>0)
      assert(contains(m2, 0), "map 4")
      assert(!contains(m2, 1), "map 5")
    }
  }

  @Rule def individualTestTimeout = new org.junit.rules.Timeout(10 * 1000)
}
