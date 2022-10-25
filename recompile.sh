#!/bin/bash

if [ ! -d build ]; then
    mkdir build
fi

cd build

cmake ../cpp; make -j10


if [ -d "/sad_lib" ]; then rm -Rf /sad_lib; fi
mkdir /sad_lib
cp *.so /sad_lib && cp ./rela/*.so /sad_lib
