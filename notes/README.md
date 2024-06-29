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
    - Unsupervised
        - Predict basic stockfish evaluation?
    - Supervised
        - 
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
- Having good heursitic metrics
- Large scale unsupervised pretraining
- Making the model as large as possible and making sure all "reasoning steps" that we can come up with could be supported

## Exusting dataset
- 1.5M moves, 15 GB for existing dataset: we should be able to get to around 100M moves.

## What Needs To Be Done
1. Download 90GB of data
2. Investigate stockfish depth-to-speed tradeoff
3. Estimate how long it would take to do pretraining on 100GB
    - Understand how long would it take to run stockfish between 0-25 moves in the future
4. Estimate per unit size of model how long it will take
    - Take into account model width (i.e. parallelism)
    - Take into account model depth (span)
5. What heuristics stockfish uses (and other chess engines use) to inform what features we use. This can help us to get a notion of how to do search in the architecture.
    - We also need to understand what range the value function is in
    - We also want to understand what other representations are used for chess bots.
6. Run the initial model and get a baseline for quality, get a notion of what quality level we should aim for

## 5. How Chess Bots Work
https://blogs.cornell.edu/info2040/2022/09/30/game-theory-how-stockfish-mastered-chess/ says that usually the range of values is between -4 and 4 for an absolute loss to an absolute win, with zero being neutral.

NOTE: stockfish even at 15 (which we estimated could take a reasonable amount of time) reaches 2.5K ELO! Look: https://chess.stackexchange.com/questions/8123/stockfish-elo-vs-search-depth. Maybe we can get really fast 2K ELO at least.

- Question: we will want to understand what the temperature ranges will be, because we can avoid making "mistakes". This is good if we make mistakes. Unclear if this is going to work.

- Question: is our valuation used in a tree search or in a 1-step move decision process?
- Question: can we use attention or other such functionality to try and encode boolean decisions in a learned search?
- Question: what sorts of representations can help?
- **KEY OBSERVATION** we should play as aligned with stockfish (standard?) as possible because the winner of the moves is the person who picks the move that moves to a board state with the best value ACCORDING TO STOCKFISH (25 looks ahead).

Other help from Claude opus.

We will want to have a large taxonomy of metrics based on:
1. Centrality
    - Some pieces are more effective in the center
    - More pieces in the center is better
2. Conflict
    - How many pieces are under attack
    - How many pieces that are under attack are defended or undefended
    - Are there any forks upcoming?
    - 
3. Material
    - How many pieces do I have?
        - Weighted by piece quality
4. Position
    - Pawns
        - Passed
        - Doubled
        - 
    - Reachability
        - How many squares can my pieces reach
        - How many open pathways are there to my king

Ideas for features to add
- Defended mask (boolean: for each piece, is it defended?)
- Numbers of pieces on certain areas
- Under attack square
- Reachable square
- Booleans
    - ???
    - Bishop pair