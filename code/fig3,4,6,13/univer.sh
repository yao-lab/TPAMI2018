#!/bin/bash
core_num=16
iter_num=20
for i in $(seq 1 $core_num)
do
  b=$(($(($iter_num-1))/$(($core_num/$i))+1))
  c=$(($(($iter_num-1))/b+1))
  j=1
  while [ $j -le $b ]
  do
    k=1
    while [ $k -le $c ]
    do
      let "l=(j-1)*c+k"
      if [ $l -le 20 ]; then
        mix_lbi/bin/parallel_lbi_without_feature mix_lbi/data/data_univer.txt 10 0.0001 $i 1 1 &
      fi
      let "k += 1"
    done
    wait
    let "j += 1"
  done
  #echo "Univer data with $i threads is Done!"
done
