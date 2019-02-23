;; Programming Languages, Homework 5

#lang racket
(provide (all-defined-out)) ;; so we can put tests in a second file

;; definition of structures for MUPL programs - Do NOT change
(struct var  (string) #:transparent)  ;; a variable, e.g., (var "foo")
(struct int  (num)    #:transparent)  ;; a constant number, e.g., (int 17)
(struct add  (e1 e2)  #:transparent)  ;; add two expressions
(struct ifgreater (e1 e2 e3 e4)    #:transparent) ;; if e1 > e2 then e3 else e4
(struct fun  (nameopt formal body) #:transparent) ;; a recursive(?) 1-argument function
(struct call (funexp actual)       #:transparent) ;; function call
(struct mlet (var e body) #:transparent) ;; a local binding (let var = e in body) 
(struct apair (e1 e2)     #:transparent) ;; make a new pair
(struct fst  (e)    #:transparent) ;; get first part of a pair
(struct snd  (e)    #:transparent) ;; get second part of a pair
(struct aunit ()    #:transparent) ;; unit value -- good for ending a list
(struct isaunit (e) #:transparent) ;; evaluate to 1 if e is unit else 0

;; a closure is not in "source" programs but /is/ a MUPL value; it is what functions evaluate to
(struct closure (env fun) #:transparent) 

;; Problem 1

;; CHANGE (put your solutions here)

;(a)
(define (racketlist->mupllist rlist)
  (if (eq? null rlist)
      (aunit)
      (apair (car rlist) (racketlist->mupllist (cdr rlist)))))

;(b)
(define (mupllist->racketlist mlist)
  (if (aunit? mlist)
      null
      (cons (apair-e1 mlist) (mupllist->racketlist (apair-e2 mlist)))))
;; Problem 2

;; lookup a variable in an environment
;; Do NOT change this function

;env is a Racket list of Racket pairs to represent this environment (which is initially empty)
(define (envlookup env str)
  (cond [(null? env) (error "unbound variable during evaluation" str)]
        [(equal? (car (car env)) str) (cdr (car env))]
        [#t (envlookup (cdr env) str)]))

;; Do NOT change the two cases given to you.  
;; DO add more cases for other kinds of MUPL expressions.
;; We will test eval-under-env by calling it directly even though
;; "in real life" it would be a helper function of eval-exp.

;an example for a item in env is (cons "foo" (int 100))
(define (eval-under-env e env)
  (cond [(var? e) 
         (envlookup env (var-string e))]
        [(add? e) 
         (let ([v1 (eval-under-env (add-e1 e) env)]
               [v2 (eval-under-env (add-e2 e) env)])
           (if (and (int? v1)
                    (int? v2))
               (int (+ (int-num v1) 
                       (int-num v2)))
               (error "MUPL addition applied to non-number")))]
        ;; CHANGE add more cases here
        ;;A mupl value is a mupl integer constant, a mupl closure, a mupl aunit, or a mupl pair of mupl values.
        [(int? e) e]
        [(closure? e) e]
        [(aunit? e) e]
        [(fun? e) (closure env e)]
        [(ifgreater? e)
         (let ([v1 (eval-under-env (ifgreater-e1 e) env)]
               [v2 (eval-under-env (ifgreater-e2 e) env)])
           (if (and (int? v1) (int? v2))
               (if (> (int-num v1) (int-num v2))
                   (eval-under-env (ifgreater-e3 e) env)
                   (eval-under-env (ifgreater-e4 e) env))
               (error "MUPL ifgreater applied to non-number")))]
        [(mlet? e)
         (let* ([eval (eval-under-env (mlet-e e) env)]
                [newEnv (cons (cons (mlet-var e) eval) env)])
           (eval-under-env (mlet-body e) newEnv))]
        [(call? e)
         (let ([fexp (eval-under-env (call-funexp e) env)]
               [sexp (eval-under-env (call-actual e) env)])
           (if (closure? fexp)
               (let* ([thisfun (closure-fun fexp)]
                      [thisenv (cons (cons (fun-formal thisfun) sexp) (closure-env fexp))]
                      [thisenv (if (fun-nameopt thisfun)
                                    (cons (cons (fun-nameopt thisfun) fexp) thisenv)
                                     thisenv)])
                      (eval-under-env (fun-body thisfun) thisenv))
               (error "MUPL call applied to non-funtion")))]
        [(apair? e)
         (let ([v1 (eval-under-env (apair-e1 e) env)]
               [v2 (eval-under-env (apair-e2 e) env)])
           (apair v1 v2))]
        [(fst? e)
         (let ([ret (eval-under-env (fst-e e) env)])
           (if (apair? ret)
               (apair-e1 ret)
               (error "MUPL fst applied to non-pair")))]
        [(snd? e)
         (let ([ret (eval-under-env (snd-e e) env)])
           (if (apair? ret)
               (apair-e2 ret)
               (error "MUPL snd applied to non-pair")))] 
        [(isaunit? e)
         (let ([ret (eval-under-env (isaunit-e e) env)])
           (if (aunit? ret)
               (int 1)
               (int 0)))]
        [#t (error (format "bad MUPL expression: ~v" e))]))

;; Do NOT change
(define (eval-exp e)
  (eval-under-env e null))
        
;; Problem 3
;(a)
(define (ifaunit e1 e2 e3)
  (ifgreater (isaunit e1) (int 0) e2 e3))

;(b)
(define (mlet* lstlst e2)
  (if (null? lstlst)
      e2
      (mlet (car (car lstlst))
            (cdr (car lstlst))
            (mlet* (cdr lstlst) e2))))

;(c)
#|
(define (ifeq e1 e2 e3 e4)
  (if (and (int? e1) (int? e2))
      (let ([_x (int-num e1)]
            [_y (int-num e2)])
        (if (= _x _y) e3 e4))
      (error "MUPL ifea applied to non-number")))
|#
(define (ifeq e1 e2 e3 e4)
  (mlet* (list (cons "_x" e1) (cons "_y" e2))
         (ifgreater (var "_x") (var "_y") e4
                    (ifgreater (var "_y") (var "_x") e4 e3))))
;; Problem 4
;the function is curried so it exactly receieve one parameter
#|
(define a (fun "a" "x"
               (ifaunit (var "x")
                   (int 0)
                   (add (fst (var "x")) (call (var "a") (snd (var "x")))))))

(eval-exp (call a (apair (int 9) (apair (int 10) (aunit)))))
|#
(define mupl-map
  (fun "mupl-map" "map-fun" 
                   (fun "helper" "ls"
                                  (ifaunit (var "ls")
                                           (aunit)
                                           (apair (call (var "map-fun") (fst (var "ls"))) (call (var "helper") (snd (var "ls"))))))))
                            
(define mupl-mapAddN 
  (mlet "map" mupl-map
        ;"CHANGE (notice map is now in MUPL scope)"
        (fun #f "val"
             (call (var "map") (fun #f "x" (add (var "x") (var "val")))))))

;; Challenge Problem

(struct fun-challenge (nameopt formal body freevars) #:transparent) ;; a recursive(?) 1-argument function

;; We will test this function directly, so it must do
;; as described in the assignment
(define (compute-free-vars e) "CHANGE")

;; Do NOT share code with eval-under-env because that will make
;; auto-grading and peer assessment more difficult, so
;; copy most of your interpreter here and make minor changes
(define (eval-under-env-c e env) "CHANGE")

;; Do NOT change this
(define (eval-exp-c e)
  (eval-under-env-c (compute-free-vars e) null))
