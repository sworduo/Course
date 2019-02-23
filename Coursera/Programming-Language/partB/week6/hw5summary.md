*	racket文件一定要保存才生效！光点击绿色那个开始键是不会保存的！如果没有保存，那么你对hw5的更改就不会显示在hw5test里，只有更改了hw5并且立即把保存后，hw5test才能看到你的更改！
*	还有注意let,let*,letrec的区别！let是只用let语句外面的变量，不会进行shadow，也就是说，如果

```racket
(let ([num 100]
      [val (+ num 20)])
      (bla bla))
````

将会报错！因为let外面并没有定义num这个变量，而在let内部的其他变量val是看不见的，val只看得见let之外的变量！  
这时候就应该使用let* 或者letrec了，let*是let内部变量会覆盖掉外部变量。  
而letrec则是可以使用letrec()中定义的所有函数，此时如果一个变量a在变量b后面定义，那如果b用的到了a，也还是会报错。但是，如果一个函数f在后面定义，那么在f定义前的其他绑定可以用到f！  
*	这个作业的problem3目的是使用racket函数，将一种MUPL表达式转换成另一种MUPL表达式，以实现类似于MUPL宏的功能。既然是转换为另一种MUPL表达式，那么就思考MUPL有哪些关键字可以使用，本质上来说，只是构建一个racket函数，生成新的MUPL的关键字a，而这个关键字a是由之前的MUPL关键字所组成的。一定要注意，返回的是MUPL表达式，而不是直接的evaluate结果！比如对于problem3 c，一开始我是这么写的：

```racket
(define (ifeq e1 e2 e3 e4)
  (if (and (int? e1) (int? e2))
      (let ([_x (int-num e1)]
            [_y (int-num e2)])
        (if (= _x _y) e3 e4))
      (error "MUPL ifea applied to non-number")))
```

然而这是不对的，因为这里并不是将一种MUPL表达式展开成另一种MUPL表达式，而是类似于直接计算这个表达式了！，应该这么写才对：

```racket
(define (ifeq e1 e2 e3 e4)
  (mlet* (list (cons "_x" e1) (cons "_y" e2))
         (ifgreater (var "_x") (var "_y") e4
                    (ifgreater (var "_y") (var "_x") e4 e3))))
```

这里我们用前面定义好的mlet*和ifgreater关键字来展开ifeq这个新的关键字。这才叫宏！  
*	一个明显的错误例子，考虑problem3 (a)，假如我们这么写：

```racket
(define (ifaunit e1 e2 e3)
	(if (aunit? e1) e2 e3))
```

这么写会出错！第一，这并不是将一个MUPL语句转换成另一个MUPL语句，因为MUPL里面没有关键字if！其次，这么做的话，当e1确实是aunit类型的时候的确work，但是如果e1是(var "x")类型的呢？此时e1有可能是aunit类型的，但是只有在其经过eval-under-env之后它 才是aunit类型的，在没有eval之前是val类型的，所以如果e1是(var "x")，那么这条语句将会无脑执行e3，e2将永远不会执行。  
正确的代码应该是这样的：

```racket
(define (ifaunit e1 e2 e3)
  (ifgreater (isaunit e1) (int 0) e2 e3))
```

*	首先MUPL里面的确有ifgreater这个关键字，其次isaunit在判断时，会首先对e1进行eval，假如e1是(var "x")对应的aunit，这时候就能正确eval，并且执行e2语句。同理还有snd和fst这两个关键字。一开始我也不理解为什么要特意搞这两个关键字，直接(apair-e1 e)不就好了。现在我明白了，这两个关键字就是用来应对(var "x")的这种情况的。这种情况一般出现在函数参数里面。  
*	同样的，在使用MUPL编写程序时，也要**只用到**MUPL**特有**的关键字，其他诸如if、let这些MUPL没有定义的关键字不能出现在用MUPL关键字编写的程序李。  

**总结**：当使用racket函数编写MUPL的宏的时候，要时刻注意，你用到的关键字是不是MUPL**特有**的？比如if，let这些MUPL里面没有的关键字，就不能出现在rancket函数体里面，因为这些不是MUPL的关键字，使用eval-exp时将不能正确估计这些值,导致最终的结果和我们预想的不符。  
*	free variables:没有在fun内部被shadow掉的外部定义的变量。比如(lambda () (+ x y z))里xyz都是free，因为他们都是外部定义的的。而(lambda (x) (let ([y 100]) (+x y z)))只有z是free，因为x的和参数，y是内部定义的。注意(lambda (x) (+ y (let ([y z]) (+ y y))))里y,z都是free，因为第一个y是外面定义的，第二个y对应的值z也是外部定义的。  

