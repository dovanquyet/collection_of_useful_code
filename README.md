
## Collection

Please search for the algorithms you need here, to get the correct links/code files in the `src` folder. Note that almost all of the code will be Pytorch. I mark the code in the format (order, DL framework), e.g (1, pt)
> Also, please install the dependency for each one that you want to use. There is no common dependency, thus there will be no file requirement.txt here

### Parallel
- training/inference with deepspeed intergration: [(1, pt)](https://github.com/tqfang/comet-deepspeed)
- easy-to-use parallel generation (modified from Pytorch built-in parallel for classification inference): [(1, pt)](./src/model_generate_parallel.py)

### Algorithm
- in parallel, calculating LM's loss for each token and take sum as score for zeroshot inference: [(1, pt)](./src/masked_lm_loss_avg_tokens.py)
- compute the score for Masked LM task: [(1, pt)](https://gist.github.com/tqfang/e6a796e650e8db70bdda2ecb7bae4317)