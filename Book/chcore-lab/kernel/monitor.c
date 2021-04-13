/*
 * Copyright (c) 2020 Institute of Parallel And Distributed Systems (IPADS), Shanghai Jiao Tong University (SJTU)
 * OS-Lab-2020 (i.e., ChCore) is licensed under the Mulan PSL v1.
 * You can use this software according to the terms and conditions of the Mulan PSL v1.
 * You may obtain a copy of Mulan PSL v1 at:
 *   http://license.coscl.org.cn/MulanPSL
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
 *   PURPOSE.
 *   See the Mulan PSL v1 for more details.
 */

// Simple command-line kernel monitor useful for
// controlling the kernel and exploring the system interactively.

#include <common/printk.h>
#include <common/types.h>

static inline __attribute__ ((always_inline))
u64 read_fp()
{
	u64 fp;
	__asm __volatile("mov %0, x29":"=r"(fp));
	return fp;
}

__attribute__ ((optimize("O1")))
int stack_backtrace()
{
	printk("Stack backtrace:\n");

	// Your code here.
	//回溯会回溯到fp的值为0的情况，也就是整个程序最开始的函数那里
	//注意fp是当前函数的栈的栈顶，这是个地址！里面的值才是我们想要的
	u64* current_fp = (u64*)read_fp();
	// printk("%lx %lx\n", current_fp, *current_fp);
	while(*current_fp){
		u64* lastfp = (u64*)*current_fp;
		u64* lr = (u64*)*((u64*)(lastfp+1));
		u64 arg = *(u64*)(current_fp+2);
		printk("LR %lx FP %lx Args %lx\n", lr, lastfp, arg);
		current_fp = lastfp;
	}
	return 0;
}