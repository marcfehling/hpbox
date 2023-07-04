#!/bin/bash

nrepetitions=1
nprocs=96

exec="/tmp/fehling/build-hpbox-int64/hprun"



for prm in *.prm
do
  I=0
  while [ ${I} -lt ${nrepetitions} ]
  do
    CMD="mpirun -np ${nprocs} ${exec} ${prm}"
    echo ${CMD}
    ${CMD}
    let I=${I}+1
  done
done
