package funsets

object Main extends App {
  import FunSets._
  println(contains(singletonSet(1), 2))
  println(contains(singletonSet(2), 2))
  println(!contains(singletonSet(1), 1))
  val a = 100
  println(-a)
}
