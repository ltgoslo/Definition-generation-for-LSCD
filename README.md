# Definition Generation for Lexical Semantic Change Detection (LSCD)

## Repository structure

    .
    ├── analysis      # APD and PRT results analysis
    ├── apd_prt       # APD and PRT experiments
    ├── embeddings    # generating definitions embeddings
    ├── gold          # ground truth, SemEval official Evaluation script
    ├── src           # Running Tang et al. (2023)'s method, see more in its README
    ├── src_acl       # generating definitions and merging them, stats etc.           

## Reproduce evaluation of LSCD performance with definition embeddings obtained with different decoding strategies (Table 3)

```
cd apd_prt
./evaluate.sh
```
