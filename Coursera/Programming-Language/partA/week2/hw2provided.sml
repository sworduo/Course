(* Dan Grossman, Coursera PL, HW2 Provided Code *)

(* if you use this function to compare two strings (returns true if the same
   string), then you avoid several of the functions in problem 1 having
   polymorphic types that may be confusing *)
fun same_string(s1 : string, s2 : string) =
    s1 = s2

(* put your solutions for problem 1 here *)

(* a *)
fun all_except_option (str, strlist) =
    case strlist of
	[] => NONE
      | curstr::strlist' => if same_string(str, curstr)
			    then SOME (strlist')
			    else
				let val prelist = all_except_option(str, strlist')
				in
				    case prelist of
					NONE => NONE
				      | SOME plist=> SOME(curstr::plist)
				end

(* b  *)
(* I have used a local val binding for no useful purpose,:( *)
fun get_substitutions1 (substitutions, s) =
    case substitutions of
	[] => []
      | firstlist::substitutions' => let val isInsert = all_except_option(s, firstlist)
				     in	case isInsert of
					    NONE => get_substitutions1(substitutions', s)
					  | SOME curlist => curlist @ get_substitutions1(substitutions', s)
				     end
				    
(* c  *)
fun get_substitutions2 (substitutions, s) =
    let fun helper(subs, s, retlist) =
	    case subs of
		[] => retlist
	      | firstlist::subs' => let val isInsert = all_except_option(s, firstlist)
					     in case isInsert of
						    NONE => helper(subs', s, retlist)
						  | SOME curlist => helper(subs', s, retlist @ curlist)
				    end
    in
	helper(substitutions, s, [])
    end

(* d  *)
fun similar_names (substitutions, {first=x, middle=y, last=z}) =
    let val namelist = get_substitutions2(substitutions, x)
	fun helper (namelist) =
	    case namelist of
		[] => []
	      | curname::namelist' => {first=curname, middle=y, last=z}::helper(namelist') 
    in
	{first=x, middle=y, last=z}::helper(namelist)
    end
    
					 
(* you may assume that Num is always used with values 2, 3, ..., 10
   though it will not really come up *)
datatype suit = Clubs | Diamonds | Hearts | Spades
datatype rank = Jack | Queen | King | Ace | Num of int 
type card = suit * rank

datatype color = Red | Black
datatype move = Discard of card | Draw 

exception IllegalMove

(* put your solutions for problem 2 here *)
(* a  *)
fun card_color thiscard =
    case thiscard of
	(Clubs, _) => Black
      | (Spades, _) => Black
      | _ => Red 

(* b  *)
fun card_value thiscard =
    case  thiscard of
	(_, Num value) => value
      | (_, Ace) => 11
      | _ => 10 

(* c  *)
fun remove_card (cs, c, e) =
    case cs of
	[] => raise e
      | fc::cs' => if c=fc then cs' else fc::remove_card(cs', c, e)

(* d  *)
(* fun all_same_color cards = *)
(*     let val color = card_color cards *)
(*     	fun is_same cards = *)
(*     	    case cards of *)
(*     		[] => true *)
(*     	      | fc::cards' => if (card_color fc) = color then is_same cards' else false *)
(*     in *)
(*     	is_same cards *)
(*     end *)
fun all_same_color cards =
    case cards of
	[] => true
      | fc::[] => true
      | fc::sc::cards' => if (card_color fc) = (card_color sc)
			  then all_same_color (sc::cards')
			  else false

(* e  *)
fun sum_cards cards =
    let fun sum (cards, acc) =
	    case cards of
		[] => acc
	      | fc::cards' => sum(cards', acc + card_value fc)
    in
	sum (cards, 0)
    end

(* f  *)
fun score (held_cards, goal) =
    let val cardsum = sum_cards held_cards
	val presum = if cardsum > goal then 3 * (cardsum-goal) else goal - cardsum
    in
	if all_same_color held_cards
	then presum div 2
	else presum
    end
    

(* g  *)
fun officiate (card_list, move_list, goal) =
    let
	fun helper(held_cards, card_list, move_list) =
	     case move_list of
		 [] => score (held_cards, goal)
	       | (Discard thecard)::move_list' => helper(remove_card (held_cards, thecard, IllegalMove), card_list, move_list')
	       | Draw::move_list' => (case card_list of
			      [] => score (held_cards, goal)
			    | fc::card_list' => if sum_cards(fc::held_cards) > goal
						then score(fc::held_cards, goal)
						else helper(fc::held_cards, card_list', move_list') 
				     )
    in
	helper ([], card_list, move_list)
    end
			     
(* Challenge Problems *)
fun score_challenge (heldcards, goal) =
    let
	(* score_sum calculate the case where all the value of Ace are set to 11, and then minus the goal *)
	val score_sum = (sum_cards heldcards) - goal
	val isSameColor = all_same_color heldcards
	(* count the number of Ace *)
	fun countAce (cards, acc) =
	    case cards of
		[] => acc
	      | (s,r)::cards' => countAce (cards', acc + (if r=Ace then 1 else 0))
	fun scoreSign x = x * (if x > 0 then 3 else ~1)
	fun finalScore x = if isSameColor then x div 2 else x
	(* try out all the possibility that the number of Ace set to 1 *)
	fun helper (numAce) =
	    case numAce of
		0 => finalScore (scoreSign score_sum)
	      (* the number of Ace that set to 1 is numAce *)
	      | numAce => let val curScore = finalScore (scoreSign (score_sum - numAce * 10))
			      val bestScore = helper(numAce - 1)
			  in
			      if curScore < bestScore then curScore else bestScore
			  end
    in
	helper(countAce (heldcards, 0))
    end
							
fun officiate_challenge (card_list, move_list, goal) =
    let
	fun helper(held_cards, card_list, move_list) =
	    case move_list of
		[] => score_challenge (held_cards, goal)
	     |  (Discard thiscard)::remain_move_list => helper(remove_card(held_cards, thiscard, IllegalMove), card_list, remain_move_list)
	     | Draw::remain_move_list => case card_list of
					     [] => score_challenge (held_cards, goal)
					   | c::remain_card_list => 
					       (* 默认ACE取最小值1，如果取了1还是超过goal才计算分数。 *)
					     if (sum_cards (c::held_cards) + (if card_value c = 11 then ~10 else 0)) > goal
					     then score_challenge (c::held_cards, goal)
					     else helper (c::held_cards, remain_card_list, remain_move_list)
							 (* 下面这么做是错误的，因为sum_cards计算时，默认Ace等于11，但是这里的Ace可能取值有两个，也就是说，当新添加的是Ace，需要从两个值来判断是否小于goal *)
							 (* the code below is wrong because in this case the value of Ace can be 11 or 1, however, the function sum_cards just compute the case where the value of Ace is always 11. So everytime we meet a Ace, we can simply consider its value is 1. *)
					     (* if sum_cards (c::held_cards) > goal *)
								    (* then score_challenge (c::held_cards, goal) *)
								    (* else helper (c::held_cards, remain_card_list, remain_move_list) *)
    in
	helper([], card_list, move_list)
    end
							 

(* careful_player *)
(* In my opinion, if we use tail recursive then we need to merge two list which means the time complexity will be high*)
(* If we not, the space comsuming is more than former*)
(* And I don't use tail recursive :) *)
fun careful_player (card_list, goal) =
    let
	(* determeter whether a card with value we want is in held_list *)
	fun isExist ([], value) = NONE
	  | isExist (c::held_list, value)  = if value = card_value c then SOME c else isExist (held_list, value)
	fun helper (held_cards, card_list) =
	    case card_list of
		[] => []
	      | c::remain_card_list => let
		  (* test if we draw the fisrt card in the card_list *)
		  val curscore = sum_cards held_cards
		  val diff = curscore + card_value c - goal
	      in
		  if curscore < goal - 10
		  then Draw::helper(c::held_cards, remain_card_list)
		  else if curscore = goal
		  then []
		  else case isExist (held_cards, diff) of
			   SOME car => (Discard car)::Draw::[]
			 | _ => if diff < 0 then helper(c::held_cards, remain_card_list) else []
	      end
    in
	helper ([], card_list)
    end
