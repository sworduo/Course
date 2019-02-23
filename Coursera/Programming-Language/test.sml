val cbs : (int -> unit) list ref = ref []
fun onKeyEvent f = cbs := f::(!cbs)
(* when "onEvent" is activated, then *)
(* 1. no matter what is passed in onEvent, the register function "fn - => timesPressed.."will be actived and increase the timesPressed one time *)
(* 1. check what is passed in, if it's an int number and equals to one of the argument registered by printifpressed, then print it *)
fun onEvent i =
    let fun loop fs =
	    case fs of
		[] => ()
	      | f::fs' => (f i; loop fs')
    in loop(!cbs)
    end

(* always activate whatever you input,so it can calculate the number the onEvent have been called *)
val timesPressed = ref 0
val _ = onKeyEvent (fn _ => timesPressed := (!timesPressed) + 1 )

(*activate the correspnding register funtion when the input satisfy the condition below i.e. the input number equals to the one of the argument registered by printIfpressed*)
fun printIfpressed i =
    onKeyEvent (fn j => if i = j
			then print ("you pressed " ^ Int.toString i ^ "\n")
			else ())

(* register three funtion with argument 4, 11 and 23, respectively *)
val _ = printIfpressed 4
val _ = printIfpressed 11
val _ = printIfpressed 23
		       
fun hh x =
    case x of
    [] => 5
  | x1::y => 6
  | x1::[] => 7 
