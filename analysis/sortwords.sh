#! /bin/sh

sort "$1"-*-js.tsv | sort -k2 -n
