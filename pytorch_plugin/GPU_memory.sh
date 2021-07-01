#!/bin/bash

target=$1

if [ -z "$target" ]
then
    echo "No argument provided. No changes made."
else
    echo 'Changing SIZE_SHARED_CACHE to '$1'.'

    ssc=$(cat CUDACachingAllcator.cpp | grep 'define SIZE_SHARED_CACHE' | cut -d' ' -f3 | cut -d'(' -f2)

    cat CUDACachingAllcator.cpp | sed -e "s/SIZE_SHARED_CACHE ($ssc/SIZE_SHARED_CACHE ($target/g" > tmp.cpp
fi