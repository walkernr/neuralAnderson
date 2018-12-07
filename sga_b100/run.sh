#!/bin/bash
python ../scripts/parse.py -v
python ../scripts/correlation.py -v
for i in 19
do
  for j in 3
  do
    python ../scripts/ml.py -v -p -pt -nt 8 -mi $i -nc $j
  done
done
