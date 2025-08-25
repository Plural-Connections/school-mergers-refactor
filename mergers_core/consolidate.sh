#!/bin/bash -e
shopt -s globstar
# consolidates all the results.csv files found within ../data/results into one large csv file
# then, deduplicates configurations to ensure that only the most recent run of a configuration is kept
#
# the output is called <batchname>.csv where batchname is the only batch in ../data/results or
# the first argument if there are multiple present

pushd ../data/results > /dev/null
batches=(*)

if [[ ${#batches[@]} -gt 1 && -z $1 ]]; then
    echo "multiple batches found: $batches"
    echo "please specify a batch name as the first argument"
    exit 1
fi
batchname=${1:-${batches[0]}}
echo "using batch $batchname"

files=($batchname/**/analytics.csv)
files=($(ls -t "${files[@]}"))
echo "header is from ${files[0]}"
head -n 1 "${files[0]}" > $batchname.csv  # get csv column names once
tail -q -n +2 "${files[@]}" | cut -d, -f -40 >> $batchname.csv  # skip csv column names

popd > /dev/null
# python3 consolidate_deduplicate.py $batchname
