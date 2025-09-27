#!/bin/bash -e
shopt -s globstar
# consolidates all the results.csv files found within ../data/results into one large csv file
# then, deduplicates configurations to ensure that only the most recent run of a configuration is kept
#
# the output is called results.csv

olddir=$PWD
cd ../data/results

files=(**/analytics.csv)
files=($(ls -t "${files[@]}"))
echo "header is from ${files[0]}"
head -n 1 "${files[0]}" > results.csv  # get csv column names once
tail -q -n +2 "${files[@]}" | cut -d, -f -29 >> results.csv  # skip csv column names and possible extra columns

python3 $olddir/consolidate_deduplicate.py
cd $olddir
