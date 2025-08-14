#!/bin/bash -e
shopt -s globstar
# consolidates all the results.csv files found within ../data/results into one large csv file
# then, deduplicates configurations to ensure that only the most recent run of a configuration is kept
#
# the output is called <batchname>.csv where batchname is the only batch in ../data/results or
# the first argument if there are multiple present

cd ../data/results
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
head -n 1 "${files[0]}" > $batchname.csv  # get csv column names once
tail -q -n +2 "${files[@]}" >> $batchname.csv  # skip csv column names

lines_pre=$(wc -l $batchname.csv | awk '{print $1}')
python3 <<EOF
import pandas as pd
df = pd.read_csv("${batchname}.csv")\
       .drop_duplicates(subset=["district_id", "state", "school_decrease_threshold", "dissimilarity_weight",
                                "population_consistency_weight", "population_consistency_metric",
                                "dissimilarity_flavor"], keep="first")
print(f"dropped {$lines_pre - len(df) - 1} line(s)")
df.to_csv("${batchname}.csv", index=False)
EOF
