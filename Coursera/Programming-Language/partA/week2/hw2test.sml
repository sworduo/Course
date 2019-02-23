(* Homework2 Simple Test *)
(* These are basic test cases. Passing these tests does not guarantee that your code will pass the actual homework grader *)
(* To run the test, add a new line to the top of this file: use "homeworkname.sml"; *)
(* All the tests should evaluate to true. For example, the REPL should say: val test1 = true : bool *)
use "hw2provided.sml";
(* val test1 = all_except_option ("string", ["string"]) = SOME [] *)
(* val test111 = all_except_option("str", ["str", "teh"]) *)
(* val test112 = all_except_option("str", ["sstr", "the"]) *)
(* val test113 = all_except_option("str", ["the", "str"]) *)
(* val test114 = all_except_option("str", ["the", "str", "sstr"]) *)

(* val test2 = get_substitutions1 ([["foo"],["there"]], "foo") = [] *)
(* val test22 = get_substitutions1([["f", "fred"],["E","Bi"],["f","fredd","ok"]],"f") *)
(* val test23 = get_substitutions1([["f", "fred"],["E","Bi"],["f","fredd","ok","fred"]],"f") *)

(* val test31 = get_substitutions1 ([["foo"],["there"]], "foo") = [] *)
(* val test32 = get_substitutions2([["f", "fred"],["E","Bi"],["f","fredd","ok"]],"f") *)
(* val test33 = get_substitutions2([["f", "fred"],["E","Bi"],["f","fredd","ok","fred"]],"f") *)

(* val test3 = get_substitutions2 ([["foo"],["there"]], "foo") = [] *)
												 
(* val test4 = similar_names ([["Fred","Fredrick"],["Elizabeth","Betty"],["Freddie","Fred","F"]], {first="Fred", middle="W", last="Smith"}) = *)
(* 	    [{first="Fred", last="Smith", middle="W"}, {first="Fredrick", last="Smith", middle="W"}, *)
(* 	     {first="Freddie", last="Smith", middle="W"}, {first="F", last="Smith", middle="W"}] *)

(* val test5 = card_color (Spades, Num 2) = Black *)

(* val test6 = card_value (Clubs, Jack) = 10 *)

(* val test7 =( remove_card ([(Hearts, Ace)], (Hearts, Jack), IllegalMove)handle IllegalMove => []) =[] *)

(* val test8 = all_same_color [(Hearts, Ace), (Hearts, Ace)] = true *)
(* val test81 = all_same_color [(Hearts,Ace), (Hearts, Ace), (Hearts, Ace)] = true *)
(* val test82 = all_same_color [(Hearts,Ace)] = true *)
(* val test83 = all_same_color [(Clubs, Jack), (Hearts, Ace)] = false *)
								

(* val test9 = sum_cards [(Clubs, Num 3),(Clubs, Jack)]  *)

(* val test10 = score ([(Spades, Num 2),(Clubs, Num 4)],10)  *)

(* val test11 = officiate ([(Hearts, Num 2),(Clubs, Num 4)],[Draw], 15) = 6 *)

(* val test12 = officiate ([(Clubs,Ace),(Spades,Ace),(Clubs,Ace),(Spades,Ace)], *)
(*                         [Draw,Draw,Draw,Draw,Draw], *)
(*                         42) *)
(*              = 3 *)

(* val test13 = ((officiate([(Clubs,Jack),(Spades,Num(8))], *)
(*                          [Draw,Discard(Hearts,Jack)], *)
(*                          42); *)
(*                false) *)
(*               handle IllegalMove => true) *)
		 
(* score([], 20) = 10; *)
(* score([(Spades, Jack), (Spades, Ace)], 20) = 1; *)
(* score([(Spades, Ace), (Spades, Ace)], 20) = 3; *)
(* score([(Spades, Ace), (Clubs, Ace)], 20) = 3; *)
(* score([(Spades, Ace), (Hearts, Ace)], 20)= 6; *)

(* (officiate([(Clubs,Jack),(Spades,Num(8))], [Draw,Discard(Hearts,Jack)], 42) handle IllegalMove => 0) = 0; *)
(* officiate([(Clubs,Ace),(Spades,Ace),(Clubs,Ace),(Spades,Ace)], [Draw,Draw,Draw,Draw,Draw], 42) = 3; *)
(* officiate([(Clubs,Ace),(Clubs,Ace),(Clubs,Ace),(Clubs,Ace)], [Draw,Draw,Draw,Draw,Draw], 42) = 3; *)
(* officiate([(Clubs,Ace),(Clubs,Ace),(Clubs,Ace),(Clubs,Ace)], [Draw,Draw,Draw], 42) = 4; *)
(* officiate([(Clubs,Ace),(Clubs,Ace),(Clubs,Ace),(Clubs,Ace)], [], 42) = 21; *)
(* officiate([(Clubs,Ace),(Clubs,Ace),(Spades,Jack),(Clubs,Ace)], [Draw,Draw,Discard(Clubs,Ace) ], 42) = 15; *)

(* officiate( [(Clubs,Num 1),(Spades, Num 2),(Clubs, Num 3),(Spades, Num 4)], [Draw,Draw,Draw,Draw, Discard (Clubs, Num 2)], 40) = 15; *)
(* officiate( [(Hearts,Num 1),(Spades, Num 2),(Clubs, Num 3),(Spades, Num 4)], [Draw,Draw,Draw,Draw, Draw], 40)=30; *)
val test10 = score_challenge ([(Spades, Num 2),(Clubs, Num 4)],10)

val test11 = officiate_challenge ([(Hearts, Num 2),(Clubs, Num 4)],[Draw], 15) = 6

val test12 = officiate_challenge ([(Clubs,Ace),(Spades,Ace),(Clubs,Ace),(Spades,Ace)],
                        [Draw,Draw,Draw,Draw,Draw],
                        42)
       

val test13 = ((officiate_challenge([(Clubs,Jack),(Spades,Num(8))],
                         [Draw,Discard(Hearts,Jack)],
                         42);
               false)
              handle IllegalMove => true)
