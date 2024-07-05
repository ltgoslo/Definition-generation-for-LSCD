# Graph-based definition merging

This directory contains scripts of an alternative definition merging approach that is not discussed in the paper. With the default parameters, it works as follows:
- Each unique definition is added as a vertex to a graph. The two periods are processed jointly, so there is a single graph for both periods.
- For each definition, a normalized version is produced (lowercasing, punctuation removal).
- Definitions whose normalized versions are identical are connected by an arc.
- Moreover, if the normalized Levenshtein distance between two definitions is lower than a predefined threshold, an arc is added between the definitions. Normalized Levenshtein distance = Absolute Levenshtein distance / max(len(def1), len(def2))
- For all connected subgraphs, the original definitions are replaced by the most frequent one of the subgraph.

This merging approach is implemented in `graph_merge.py`. It has the following parameters:
- `--defgen_path`: Directory with two files containing the generated definitions.
- `--lang`: Language/task name (e.g. `norwegian1`).
- `--out_path`: Directory to save datasets with merged definitions.
- `--leven_threshold`: Levenshtein threshold - only merge definitions below threshold. Uses relative Levenshtein distance with values <1 and absolute Levenshtein distance with values >=1. Set to <0 to skip the Levenshtein-distance merging step.
- `--min_subseq_len`: Minimal length of the definition (in words) to be taken into account for subsequence matching. This is an additional additional merging step before Levenshtein-distance merging, and can be skipped by setting to <0.
- `--separate`: Process both time periods separately (joint processing if parameter is omitted).
- `--hub_strategy`: Determines how to select the target definition of a subgraph, either the most frequent one (`maxFreq`) or the one of the best connected vertex (`maxDegree`).

The script `run_example.sh` shows how to run this method with sensible (but not necessarily optimal) parameter settings.

## Results

The table below compares the correlation scores of the methods presented in the paper with the graph-based definition merging.

| Method      | English | Norwegian 1 | Norwegian 2 | Russian 1 | Russian 2 | Russian 3 | 
| ----------- | ------- | ----------- | ----------- | --------- | --------- | --------- |
| XLM-R token emb. (Table 2)     | 0.514 | 0.394 | 0.387 | 0.376 | 0.480 | 0.457 |
| Definition emb. (Table 2)      | **0.637** | 0.496 | **0.565** | 0.488 | 0.462 | 0.504 |
| No merging (Table 4)           | 0.405 | 0.332 | 0.232 | 0.390 | 0.427 | 0.469 |
| Minimalist merging (Table 4)   | *0.565* | 0.280 | 0.197 | 0.391 | 0.431 | 0.491 |
| Full-fledged merging (Table 4) | 0.418 | 0.362 | 0.260 | 0.391 | 0.416 | 0.476 |
| Graph-based merging            | 0.483 | ***0.548*** | *0.511* | ***0.559*** | ***0.610*** | ***0.620*** |

(Best overall score in bold, best explainable score in italics.)

Note that for all merging-based approaches, we use definitions generated with greedy decoding and JS divergence as a distance measure.

The graph-based merging outperforms the merging strategy defined in the paper on all languages but English. It also outperforms the non-interpretable approaches on four out of six tasks.
