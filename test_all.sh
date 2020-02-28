#!/bin/bash

models=("conv" "shufflenet"  "mixnet")
batch_sizes=(32 64 128)
criterions=("ce" "fl")

for model in ${models[*]}; do
  for batch_size in ${batch_sizes[*]}; do
    for criterion in ${criterions[*]}; do
      python3 main.py -a $model -b $batch_size -c $criterion -op sgd -lr 0.01 --normalise
      python3 main.py -a $model -b $batch_size -c $criterion -op adam -lr 0.001
    done
  done
done
