#!/bin/bash

echo "500 ps simulation"
python ../convergence_analysis.py -d ../test/data/short_dvdl.dat -t 2.0 -v -p short_example.png

echo
echo "9.5 ns simulation"
python ../convergence_analysis.py -d ../test/data/long_dvdl.dat -t 0.5 -v -p long_example.png