#! /bin/sh

echo "** CORPUS 1 **"
zcat "$1"-*-corpus1.tsv.gz | grep -P "^$2\t" | cut -f 3 | sort | uniq -c | sort -k1 -nr | head -n5
echo "Total lines:"
zcat "$1"-*-corpus1.tsv.gz | grep -P "^$2\t" | wc -l
echo ""
echo "** CORPUS 2 **"
zcat "$1"-*-corpus2.tsv.gz | grep -P "^$2\t" | cut -f 3 | sort | uniq -c | sort -k1 -nr | head -n5
echo "Total lines:"
zcat "$1"-*-corpus2.tsv.gz | grep -P "^$2\t" | wc -l
echo ""
