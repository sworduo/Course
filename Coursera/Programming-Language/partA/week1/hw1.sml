(* 1. Write a function is_older that takes two dates and evaluates to true or false. It evaluates to true if the rst argument is a date that comes before the second argument. (If the two dates are the same, the result is false.) *)
fun is_older(right : int * int * int, left : int * int * int)=
    if (#1 right < #1 left) orelse ((#1 right = #1 left) andalso (#2 right < #2 left)) orelse ((#1 right = #1 left) andalso (#2 right = #2 left) andalso (#3 right < #3 left))
    then true
    else false

(* 2. Write a function number_in_month that takes a list of dates and a month (i.e., an int) and returns how many dates in the list are in the given month. *)
fun number_in_month(dates : (int * int * int) list, month : int)=
    if null dates
    then 0
    else
	let
	    val cnt = number_in_month(tl dates, month)
	in
	    if #2 (hd dates) = month
	    then cnt + 1
	    else cnt
	end

(* 3. Write a function number_in_months that takes a list of dates and a list of months (i.e., an int list)and returns the number of dates in the list of dates that are in any of the months in the list of months.Assume the list of months has no number repeated. *)
fun number_in_months(dates : (int * int * int) list, months : int list)=
    if null dates orelse null months
    then 0
    else
	let
	    val cumsum = number_in_months(dates, tl months)
	    val cur_sum = number_in_month(dates, hd months)
	in
	    cumsum+cur_sum
	end

(* 4. Write a function dates_in_month that takes a list of dates and a month (i.e., an int) and returns a list holding the dates from the argument list of dates that are in the month. The returned list should contain dates in the order they were originally given. *)

fun dates_in_month(dates : (int * int * int) list, month : int)=
    if null dates
    then []
    else
	let
	    val prelist = dates_in_month(tl dates, month)
	in
	    if #2 (hd dates) = month
	    then (hd dates) :: prelist
	    else prelist
	end

(* 5. Write a function dates_in_months that takes a list of dates and a list of months (i.e., an int list) and returns a list holding the dates from the argument list of dates that are in any of the months in the list of months. Assume the list of months has no number repeated. Hint: Use your answer to the previous problem and SML's list-append operator (@). *)
fun dates_in_months(dates : (int * int * int) list, months : int list)=
    if null dates orelse null months
    then []
    else
	let
	    val prelist = dates_in_months(dates, tl months)
	    val curlist = dates_in_month(dates, hd months)
	in
	    curlist @ prelist
	end

(* 6. Write a function get_nth that takes a list of strings and an int n and returns the nth element of the list where the head of the list is 1st. Do not worry about the case where the list has too few elements: your function may apply hd or tl to the empty list in this case, which is okay. *)
fun get_nth(str :  string list, n : int)=
    if n = 1
    then hd str
    else get_nth(tl str, n-1)

(* 7. Write a function date_to_string that takes a date and returns a string of the form January 20, 2013(for example). Use the operator ^ for concatenating strings and the library function Int.toString for converting an int to a string. For producing the month part, do not use a bunch of conditionals. Instead, use a list holding 12 strings and your answer to the previous problem. For consistency, put a comma following the day and use capitalized English month names: January, February, March, April, May, June, July, August, September, October, November, December. *)
fun date_to_string(date : int * int * int)=
    let
	val str_month = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    in
	get_nth(str_month, #2 date)^" "^Int.toString(#3 date)^", "^Int.toString(#1 date)
    end


(* 8.Write a function number_before_reaching_sum that takes an int called sum, which you can assume is positive, and an int list, which you can assume contains all positive numbers, and returns an int. You should return an int n such that the rst n elements of the list add to less than sum, but the rst n+1 elements of the list add to sum or more. Assume the entire list sums to more than the passed in value; it is okay for an exception to occur if this is not the case. *)
(* what about if even the first element in nums is bigger than sum, in this case should I easily return 0? *)
fun number_before_reaching_sum(sum : int, nums : int list)=
    if hd nums >= sum
    then 0
    else 1 + number_before_reaching_sum(sum - hd nums, tl nums)
				       
(* 9.Write a function what_month that takes a day of year (i.e., an int between 1 and 365) and returns what month that day is in (1 for January, 2 for February, etc.). Use a list holding 12 integers and your answer to the previous problem. *)
fun what_month(date : int)=
    let
	val days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	val month = number_before_reaching_sum(date, days)				      
    in
	month + 1
    end

(* 10. Write a function month_range that takes two days of the year day1 and day2 and returns an int list [m1,m2,...,mn] where m1 is the month of day1, m2 is the month of day1+1, ..., and mn is the month of day day2. Note the result will have length day2 - day1 + 1 or length 0 if day1>day2. *)
fun month_range(day1 : int, day2 : int)=
    if day1 > day2
    then []
    else if day1 = day2
    then [what_month(day2)]
    else
	let
	    val cur_month = what_month(day1)
	    val pre_months = month_range(day1 + 1, day2)
	in
	    cur_month :: pre_months
	end
	
(* 11. Write a function oldest that takes a list of dates and evaluates to an (int*int*int) option. It evaluates to NONE if the list has no dates and SOME d if the date d is the oldest date in the list. *)
fun oldest(dates : (int * int * int) list)=
    if null dates
    then NONE
    else
	let
	    val last_oldest = oldest(tl dates)
	in
	    if last_oldest = NONE orelse is_older(hd dates, valOf(last_oldest))
	    then SOME (hd dates)
	    else last_oldest
	end

(* A helper function for challenge problem.*)
fun isExist(month : int, months : int list)=
    if null months
    then false
    else if month = hd months
    then true
    else
	isExist(month, tl months)
	
fun helper(months : int list)=
    if null months
    then []
    else
	let
	    val UniMonth = helper(tl months)
	in
	    if isExist(hd months, UniMonth)
	    then UniMonth
	    else (hd months) :: UniMonth
	end
	    
(* 12. Challenge Problem: Write functions number_in_months_challenge and dates_in_months_challenge that are like your solutions to problems 3 and 5 except having a month in the second argument multiple times has no more eect than having it once. (Hint: Remove duplicates, then use previous work.) *)
fun number_in_months_challenge(dates : (int * int * int) list, months : int list)=
    if null dates orelse null months
    then 0
    else
	number_in_months (dates, helper(months))

fun dates_in_months_challenge(dates : (int *int * int) list, months : int list)=
    if null dates orelse null months
    then []
    else
	dates_in_months (dates, helper(months))

(* 13. Challenge Problem: Write a function reasonable_date that takes a date and determines if i describes a real date in the common era. A \real date" has a positive year (year 0 did not exist), a month between 1 and 12, and a day appropriate for the month. Solutions should properly handle leap years. Leap years are years that are either divisible by 400 or divisible by 4 but not divisible by 100. (Do not worry about days possibly lost in the conversion to the Gregorian calendar in the Late 1500s.) *)
fun reasonable_date(date : int * int * int)=
    let
	fun isleap(year : int)=
	    if (year mod 400 = 0) orelse (year mod 4 = 0 andalso year mod 100 <>0)
	    then true
	    else false

	val months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	(* get the length of a month *)
	fun getmonth(month : int, months : int list)=
	    if month = 1
	    then hd months
	    else getmonth(month-1, tl months)
	
    in
	if ((#1 date) <= 0) orelse ((#2 date) < 1) orelse ((#2 date) > 12) orelse ((#3 date) < 1)
	then false
	else
	    let
		val month_length = getmonth(#2 date, months)
	    in
		if isleap(#1 date) andalso ((#2 date) = 2)
		then
		    if (#3 date) > month_length + 1
		    then false
		    else true
		else
		    if (#3 date) > month_length
		    then false
		    else true
	    end
    end
				       
    
    
