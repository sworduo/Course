package objsets

import TweetReader._

/**
 * A class to represent tweets.
 */
class Tweet(val user: String, val text: String, val retweets: Int) {
  override def toString: String =
    "User: " + user + "\n" +
    "Text: " + text + " [" + retweets + "]"
}

/**
 * This represents a set of objects of type `Tweet` in the form of a binary search
 * tree. Every branch in the tree has two children (two `TweetSet`s). There is an
 * invariant which always holds: for every branch `b`, all elements in the left
 * subtree are smaller than the tweet at `b`. The elements in the right subtree are
 * larger.
 *
 * Note that the above structure requires us to be able to compare two tweets (we
 * need to be able to say which of two tweets is larger, or if they are equal). In
 * this implementation, the equality / order of tweets is based on the tweet's text
 * (see `def incl`). Hence, a `TweetSet` could not contain two tweets with the same
 * text from different users.
 *
 *
 * The advantage of representing sets as binary search trees is that the elements
 * of the set can be found quickly. If you want to learn more you can take a look
 * at the Wikipedia page [1], but this is not necessary in order to solve this
 * assignment.
 *
 * [1] http://en.wikipedia.org/wiki/Binary_search_tree
 */
abstract class TweetSet extends TweetSetInterface {

  /**
   * This method takes a predicate and returns a subset of all the elements
   * in the original set for which the predicate is true.
   *
   * Question: Can we implment this method here, or should it remain abstract
   * and be implemented in the subclasses?
   */
  def filter(p: Tweet => Boolean): TweetSet = filterAcc(p, new Empty)

  /**
   * This is a helper method for `filter` that propagetes the accumulated tweets.
   */
  def filterAcc(p: Tweet => Boolean, acc: TweetSet): TweetSet

  /**
   * Returns a new `TweetSet` that is the union of `TweetSet`s `this` and `that`.
   *
   * Question: Should we implment this method here, or should it remain abstract
   * and be implemented in the subclasses?
   */
  def union(that: TweetSet): TweetSet

  def head: Tweet
  def tail: TweetSet

  /**
   * Returns the tweet from this set which has the greatest retweet count.
   *
   * Calling `mostRetweeted` on an empty set should throw an exception of
   * type `java.util.NoSuchElementException`.
   *
   * Question: Should we implment this method here, or should it remain abstract
   * and be implemented in the subclasses?
   */
  def mostRetweeted: Tweet

  /**
   * Returns a list containing all tweets of this set, sorted by retweet count
   * in descending order. In other words, the head of the resulting list should
   * have the highest retweet count.
   *
   * Hint: the method `remove` on TweetSet will be very useful.
   * Question: Should we implment this method here, or should it remain abstract
   * and be implemented in the subclasses?
   */
  def descendingByRetweet: TweetList

  def isEmpty: Boolean

  /**
   * The following methods are already implemented
   */

  /**
   * Returns a new `TweetSet` which contains all elements of this set, and the
   * the new element `tweet` in case it does not already exist in this set.
   *
   * If `this.contains(tweet)`, the current set is returned.
   */
  def incl(tweet: Tweet): TweetSet

  /**
   * Returns a new `TweetSet` which excludes `tweet`.
   */
  def remove(tweet: Tweet): TweetSet

  /**
   * Tests if `tweet` exists in this `TweetSet`.
   */
  def contains(tweet: Tweet): Boolean

  /**
   * This method takes a function and applies it to every element in the set.
   */
  def foreach(f: Tweet => Unit): Unit
}

class Empty extends TweetSet {

  def filterAcc(p: Tweet => Boolean, acc: TweetSet): TweetSet = acc

  def union(that: TweetSet): TweetSet = that

  def mostRetweeted: Tweet = throw new java.util.NoSuchElementException

  def descendingByRetweet: TweetList = Nil

  def isEmpty: Boolean = true

  def head: Tweet = throw new Exception("Empty.head")
  def tail: TweetSet = throw new Exception("Empty.tail")
  /**
   * The following methods are already implemented
   */

  def contains(tweet: Tweet): Boolean = false

  def incl(tweet: Tweet): TweetSet = new NonEmpty(tweet, new Empty, new Empty)

  def remove(tweet: Tweet): TweetSet = this

  def foreach(f: Tweet => Unit): Unit = ()
}

class NonEmpty(elem: Tweet, left: TweetSet, right: TweetSet) extends TweetSet {

  def head: Tweet = if(left.isEmpty) elem else left.head

//直接remove(head)会多很多比较的过程
  def tail: TweetSet =
    if(left.isEmpty) right
    else new NonEmpty(elem, left.tail, right)


  def filterAcc(p: Tweet => Boolean, acc: TweetSet): TweetSet =
    if(p(elem)) right.filterAcc(p, left.filterAcc(p, acc)).incl(elem)
    else right.filterAcc(p, left.filterAcc(p, acc))

  def union(that: TweetSet): TweetSet = this.tail.union(that.incl(this.head))

  def isEmpty: Boolean = false

  //TODO搞错了，这里每个string需要找到其retreet来进行比较，而不是单纯的找到最右边元素（只能代表字母顺序最大，不代表retreet max)
//  def mostRetweeted: Tweet =
//    try{
//      val l = mostRetweeted(left)
//    }catch {
//      case e: java.util.NoSuchElementException => elem
//    }

  //  def descendingByRetweet: TweetList ={
//    def helper(t: Tweet, xs: TweetList): TweetList =
//      try{
//        val mostRT = mostRetweeted
//        helper(remove(mostRT), new Cons(xs, mostRT))
//      }catch {
//        case e: java.util.NoSuchElementException => new Cons(xs, Nil)
//      }
//    helper(elem)
//  }

  def mostRetweeted: Tweet = {
    //    val lmost: Tweet = Nil
    //    val rmost: Tweet = Nil
    //    if(!left isEmpty) lmost = left mostRetweeted
    //    if(!right isEmpty) rmost = right mostRetweeted
    //    if(lmost==Nil && rmost==Nil) elem
    //    else if(lmost==Nil)
    //      if(elem.retweets > rmost.retweets) elem
    //      else rmost
    //    else if(rmost==Nil)
    //      if(elem.retweets > lmost.retweets) elem
    //      else lmost
    //    else{
    //      val lrmost = rmost.retweets > lrmost.retweets ? rmost : lmost
    //      if(lrmost.retweets > elem.retweets) lrmost
    //      else elem
    //    }
    def helper(ts: TweetSet, cur: Tweet): Tweet = {
      if (ts.isEmpty) cur
      else if (ts.head.retweets < cur.retweets) helper(ts.tail, ts.head)
      else helper(ts.tail, cur)
    }
    helper(this.tail, this.head)
  }

  def descendingByRetweet: TweetList = {
    def helper(ts: TweetSet, xs: TweetList): TweetList ={
      if(ts.isEmpty) xs
      else{
        val mrt = ts.mostRetweeted
        helper(ts.remove(mrt), new Cons(mrt, xs))
      }
    }
    helper(this, Nil)
  }


  /**
   * The following methods are already implemented
   */

  def contains(x: Tweet): Boolean =
    if (x.text < elem.text) left.contains(x)
    else if (elem.text < x.text) right.contains(x)
    else true

  def incl(x: Tweet): TweetSet = {
    if (x.text < elem.text) new NonEmpty(elem, left.incl(x), right)
    else if (elem.text < x.text) new NonEmpty(elem, left, right.incl(x))
    else this
  }

  def remove(tw: Tweet): TweetSet =
    if (tw.text < elem.text) new NonEmpty(elem, left.remove(tw), right)
    else if (elem.text < tw.text) new NonEmpty(elem, left, right.remove(tw))
    else left.union(right)

  def foreach(f: Tweet => Unit): Unit = {
    f(elem)
    left.foreach(f)
    right.foreach(f)
  }
}

trait TweetList {
  def head: Tweet
  def tail: TweetList
  def isEmpty: Boolean
  def foreach(f: Tweet => Unit): Unit =
    if (!isEmpty) {
      f(head)
      tail.foreach(f)
    }
}

object Nil extends TweetList {
  def head = throw new java.util.NoSuchElementException("head of EmptyList")
  def tail = throw new java.util.NoSuchElementException("tail of EmptyList")
  def isEmpty = true
}

class Cons(val head: Tweet, val tail: TweetList) extends TweetList {
  def isEmpty = false
}


object GoogleVsApple {
  val google = List("android", "Android", "galaxy", "Galaxy", "nexus", "Nexus")
  val apple = List("ios", "iOS", "iphone", "iPhone", "ipad", "iPad")

  lazy val googleTweets: TweetSet = TweetReader.allTweets.filter(t=>google.exists(s=>t.text.contains(s)))
  lazy val appleTweets: TweetSet = TweetReader.allTweets.filter(t=>apple.exists(s=>t.text.contains(s)))

  /**
   * A list of all tweets mentioning a keyword from either apple or google,
   * sorted by the number of retweets.
   */
  lazy val trending: TweetList = googleTweets.union(appleTweets).descendingByRetweet
}

object Main extends App {
  // Print the trending tweets

  GoogleVsApple.trending foreach println
//  val set1 = new Empty
//  val set2 = set1.incl(new Tweet("a", "a body", 20))
//  val set3 = set2.incl(new Tweet("b", "b body", 20))
//  val c = new Tweet("c", "c body", 7)
//  val d = new Tweet("d", "d body", 9)
//  val set4c = set3.incl(c)
//  val set5 = set4c.incl(d)
//  set5.foreach(t=>print(t.toString+"\n"))
//  print("\nhi\n")
//  val m1 = set5.mostRetweeted
//  print(m1.toString+"\n")
//  val set6 = set5.remove(m1)
//  val m2 = set6.mostRetweeted
//  print(m2.toString+"\n")
//  val set7 = set6.remove(m2)
//  val m3 = set7.mostRetweeted
//  print(m3.toString+"\n")
//  val set8 = set7.remove(m3)
//  val m4 = set8.mostRetweeted
//  print(m4.toString+"\n")
  //val m5 = set8.remove(m4).mostRetweeted

  //set5.descendingByRetweet.foreach(t=>print(t.toString, "\n"))

}
