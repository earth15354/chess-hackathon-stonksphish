# High Leval Approaches of Attack (Ideation Part 1)
Here we are ideating how to try and improve our model. Generally the different ingredients we can try to get an edge over other teams are four-fold:
1. We can try to add better data (more datasets, etc...)
    - We could collect new datasets
    - We could augment existing datasets (this might be easier in the later game with symmetries)
2. We can use better preprocessing
    - We can add better features (for example, different integers represent, right now, whether or not some of the pieces have moved, such as the king or queen; it may be beneficial to note other such information; for example, we can add some sort of feature to represent metrics that we know are likely to be good).
    - We can use better positional or otherwise encodings to aide in the neural "reasoning"
    - The stockfish scores might be in bad units
3. Architecture (unclear what architectural changes, but the heuristic IMO is to make it very large)
4. Training
    - We could regularization of some form. Not clear what to do here exactly.
        - Regularization loss functions
        - Dropouts and other such architectural features
    - We could use a better loss function
    - We could do self-play
    - We could try learning in some curriculumn: for example, we would do some form of unsupervised representation learning, followed by some form of supervised finetuning.
        - We could do autoregression
        - We could do some sort of contrastive loss between games where the same person won, or something along those lines
    - We could hyperparameter tune

## What is likely most promising (to begin with)
- Collecting more datasets to get one more OOM
- 