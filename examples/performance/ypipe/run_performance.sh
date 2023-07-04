#!/bin/bash

nrepetitions=1
nprocs=96

exec="/raid/fehling/build-hpbox/hprun_resetfe"

directories=("${PWD}" "${PWD}/h_from_start" "${PWD}/h_from_checkpoint")



for dir in "${directories[@]}"
do
  cd ${dir}
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
done
