#!/bin/bash

# usage:
# >>> ./test.sh 0{1..9} 1{0..6}

make

for i in "$@"; do
    echo "trace$i.txt"
    ./sdriver.pl -t "trace$i.txt" -s ./tsh -a "-vp" > 1.txt &
    ./sdriver.pl -t "trace$i.txt" -s ./tshref -a "-vp" > 2.txt &
    wait
    wait
    sed -i -E 's/[0-9]{3}/123/g' 1.txt &
    sed -i -E 's/[0-9]{3}/123/g' 2.txt &
    wait
    wait
    diff 1.txt 2.txt
    echo "$i" "Done"
done
