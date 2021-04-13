cmake_minimum_required(VERSION 3.14)

SET(CMAKE_BUILD_TYPE "Debug")

add_definitions("-g")

add_library(${PROJECT_NAME}-arch OBJECT
    head.S
    main.c
    monitor.c
    common/tools.S
    common/uart.c
    common/printk.c
)
