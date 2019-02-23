(* Homework3 Simple Test*)
(* These are basic test cases. Passing these tests does not guarantee that your code will pass the actual homework grader *)
(* To run the test, add a new line to the top of this file: use "homeworkname.sml"; *)
(* All the tests should evaluate to true. For example, the REPL should say: val test1 = true : bool *)

use "hw3.sml";

(* val test1 = only_capitals ["A","B","C"] = ["A","B","C"] *)

(* val test2 = longest_string1 ["A","bc","C"] = "bc" *)
(* val test22 = longest_string1 ["ABD", "ABC", "CD"] = "ABD" *)
(* val test23 = longest_string1 [] *)
(* val test24 = longest_string1 ["aaa"] *)
			     
(* val test3 = longest_string2 ["A","bc","C"] = "bc" *)
(* val test42 = longest_string3 ["ABD", "ABC", "CD"] = "ABD" *)
(* val test43 = longest_string3 [] *)
(* val test44 = longest_string3 ["aaa"] *)
(* val test45 = longest_string4 ["ABD", "ABC", "CD"] = "ABC" *)
(* val test46 = longest_string4 [] *)
(* val test47 = longest_string4 ["aaa"] *)
			     
(* val test4a = longest_string3 ["A","bc","C"] = "bc" *)

(* val test4b = longest_string4 ["A","B","C"] = "C" *)

(* val test5 = longest_capitalized ["A","bc","C"] = "A" *)
(* val test51 = longest_capitalized ["ab","cd"] = "" *)
(* 						   val test52 = longest_capitalized ["ABC", "DDD"] = "ABC" *)

(* val test6 = rev_string "abc" = "cba" *)
(* val test61 = rev_string "" *)

(* val test7 = first_answer (fn x => if x > 3 then SOME x else NONE) [1,2,3,4,5] = 4 *)
(* val test71 = (first_answer (fn x => if x > 3 then SOME x else NONE) [1,2,2])handle NoAnswer => 666 *)

(* val test8 = all_answers (fn x => if x > 1 then SOME [x] else NONE) [2,3,4,5,6,7] *)
(* val test81 = all_answers (fn x => if x = 100 then SOME [10000] else NONE) [100,2,3] *)
(* val test82 = all_answers (fn x => if x = 100 then SOME [22] else NONE) [] *)
						  

(* val test9a = count_wildcards Wildcard = 1 *)
(* val test91 = count_wildcards (ConstructorP ("d", Wildcard)) *)
(* val test92 = count_wildcards (TupleP [Wildcard, Wildcard, ConstructorP ("a", Wildcard), TupleP [UnitP,Variable "haha"], TupleP [Wildcard, UnitP]]) *)

(* val test9b = count_wild_and_variable_lengths (Variable("a")) = 1 *)
(* val test9b2 = count_wild_and_variable_lengths (TupleP [Wildcard, Variable("abcd"), ConstructorP("bbb",Variable("dd"))] ) *)

(* val test9c = count_some_var ("x", Variable("x")) = 1 *)
(* val test9c2 = count_some_var ("x", TupleP [Wildcard, Variable("x"), Variable("y"), ConstructorP("bbb", Variable("x"))]) *)

(* val test10 = check_pat (Variable("x")) = true *)
(* val test101 = check_pat (TupleP [Variable("x"), Wildcard, ConstructorP("x", Variable("y"))]) = true *)

(* val test11 = match (Const(1), UnitP) = NONE *)

(* val test12 = first_match Unit [UnitP] = SOME [] *)


val test13 = typecheck_patterns ([("ha","DATATYPE",UnitT)], [ConstructorP("ha",UnitP), Variable("dui"), Wildcard])

val test14 = typecheck_patterns([],[TupleP[Wildcard,Wildcard], TupleP[Variable("x"),Variable("y")]])
