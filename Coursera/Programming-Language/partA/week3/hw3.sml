(* Coursera Programming Languages, Homework 3, Provided Code *)

(* problemm 1-3 should be written in val binding version *)
(* in problem 2 and 3, there also exist a pattern match about the empty-list case inside function foldl , so it's not necessary to do that by ourselves*)
fun only_capitals strlist =
    List.filter (fn c => Char.isUpper(String.sub(c, 0))) strlist

(* f(x,init) *)
fun longest_string1 strlist =
    case strlist of
	[] => ""
      | _ => foldl (fn (x, init) => if String.size init >= String.size x then init else x) "" strlist

fun longest_string2 strlist =
    case strlist of
	[] => ""
      | _ => foldl (fn (x, init) => if String.size x >= String.size init then x else init) "" strlist

fun longest_string_helper f strlist = foldl (fn (a,b) => if f(String.size a, String.size b) then a else b) "" strlist
    (* case strlist of *)
    (* 	[] => "" *)
    (*   | s::strlist' => let val longStr = longest_string_helper f strlist' *)
    (* 		      in *)
    (* 			  if f(String.size s, String.size longStr) *)
    (* 			  then s *)
    (* 			  else longStr *)
    (* 		      end *)

val longest_string3 = longest_string_helper (fn (x,y) => x > y)
val longest_string4 = longest_string_helper (fn (x,y) => x >= y)

val longest_capitalized = longest_string1 o only_capitals


val rev_string = implode o rev o explode
    
exception NoAnswer

fun first_answer f xs =
    case xs of
	[] => raise NoAnswer
      | x::xs' => case f x of
		      NONE => first_answer f xs'
		    | SOME v => v

fun all_answers f xs =
    let fun helper acc xs =
	case xs of
	    [] => SOME acc
	  | x::xs' => case (f x) of
			  SOME v => helper (v@acc) xs'
			| NONE => NONE
    in helper [] xs
    end
		 
datatype pattern = Wildcard
		 | Variable of string
		 | UnitP
		 | ConstP of int
		 | TupleP of pattern list
		 | ConstructorP of string * pattern

datatype valu = Const of int
	      | Unit
	      | Tuple of valu list
	      | Constructor of string * valu

fun g f1 f2 p =
    let 
	val r = g f1 f2 
    in
	case p of
	    Wildcard          => f1 ()
	  | Variable x        => f2 x
	  | TupleP ps         => List.foldl (fn (p,i) => (r p) + i) 0 ps
	  | ConstructorP(_,p) => r p
	  | _                 => 0
    end

val count_wildcards = g (fn _ => 1) (fn _ => 0)
(* string.size is function itself, so it's not necessary to make an anonymous-function to wrap it which exactly is f x => g x *)
val count_wild_and_variable_lengths = g (fn _ => 1) (fn x => String.size x)

fun count_some_var (s, p) =
    g (fn _ => 0) (fn x => if x = s then 1 else 0) p

fun check_pat p =
    let
	fun extractStr p =
	    case p of
		Wildcard          => []
	      | Variable x        => [x]
	      | TupleP ps         => List.foldl (fn (p, retlist) => (extractStr p) @ retlist)  [] ps
	      | ConstructorP(_,p) => extractStr p
	      | _                 => []

	fun isDistinct strlist =
	    case strlist of
		[] => true
	      | s::strlist' => if List.exists (fn x => x = s) strlist'
			       then false
			       else isDistinct strlist'
					       (* just "(List.exists (fnx=>x=s) strlist') andalso (isDistinct strlist')" maybe OK, but in which case the funtion will no longer be a tail recursive function*)
					       (* Well,in this problem tail recursive function may not be necessary.Or maybe "andalso" itself is a tail recursive function. Because it check its second condition only when the first condition is true. *)
    in
	isDistinct (extractStr p)
    end

fun match (v, p) =
    case (v,p) of
	(_,Wildcard) => SOME []
      | (_,Variable s) => SOME [(s,v)]
      | (Unit, UnitP) => SOME []
      | (Const n1, ConstP n2) => if n1 = n2 then SOME [] else NONE
      | (Tuple vs, TupleP ps) => if List.length vs = List.length ps 
				 then all_answers match (ListPair.zip(vs, ps))
				 else NONE
      | (Constructor(s1,p1), ConstructorP(s2,p2))  => if s1=s2 then match (p1, p2) else NONE
      | _ => NONE 
					


	
fun first_match v ps =
    (SOME (first_answer (fn p => match (v, p)) ps)) handle NoAnswer => NONE

	
(**** for the challenge problem only ****)

datatype typ = Anything
	     | UnitT
	     | IntT
	     | TupleT of typ list
	     | Datatype of string

(**** you can put all your code here ****)
(* It fail to pass all the test case:( *)
(* find pattern p 's type *)
fun get_pattern_type type_list p =
    let
	(* just for tuple *)
	fun helper ps =
	    case ps of
		[] => []
	      | p::ps' => (get_pattern_type type_list p)::(helper ps')
	(* just for constructorP that have typ of Datatype *)
	(* if constructorP (constructor1, type1) *)
	(* and one type in typelist have (constructor2, datatype, type2 *)
	(* if constructor1=constructor2 and type1=type2 *)
	(* then construtorP 's type is datatype *)
	fun get_datatype_type (constructorName, Datatypelist, typeName) =
	    case Datatypelist of
		(cons, datatypeName, typ)::Datatypelist' => if cons = constructorName andalso typeName = typ
							    then Datatype datatypeName
							    else get_datatype_type (constructorName, Datatypelist', typeName)
	     | [] => raise NoAnswer 
    in						   
	case p of
	    Wildcard => Anything
	  | Variable s => Anything
	  | UnitP => UnitT
	  | ConstP n => IntT
	  | TupleP ps => TupleT (helper ps)
	  | ConstructorP (s1,v) => get_datatype_type (s1, type_list, get_pattern_type type_list v)
    end


(* find the lenient type between two typ *)
fun get_lenient_type (t1,t2) =
    let
	fun helper (l1, l2) =
	    case (l1, l2) of
		([], []) => []
	      | (x::l1', y::l2') => (get_lenient_type (x,y))::helper(l1',l2')
	      | _ => raise NoAnswer
    in
	case (t1, t2) of 
	    (Anything, _) => t2
	  | (_, Anything) => t1
	  | (UnitT, UnitT) => UnitT
	  | (IntT, IntT) => IntT
	  | (Datatype s1, Datatype s2) => if s1=s2 then t1 else raise NoAnswer
	  | (TupleT l1, TupleT l2) => TupleT (helper (l1, l2))
	  | _ => raise NoAnswer
    end

(* first, find the lenient type between the first two elements of pattern list, name it cur_lenient_p*)
(* then find the lenient type between the third element and the cur_lenient_p, update cur_lenient_p *)
(* repeat the two step above *)
fun typecheck_patterns (type_list, plist) =
    SOME (foldl (fn (p, cur_lenient_p) => get_lenient_type((get_pattern_type type_list p), cur_lenient_p)) Anything plist) handle NoAnswer => NONE
				    
