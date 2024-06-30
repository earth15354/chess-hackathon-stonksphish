# Agenda
Here's what I think is important to do now:
1. Get a lot of reasonable valuation functions (maybe around 100) and make sure that they are tested (insofar as they will all return tensors of the right shape and be usable).
2. Measure (and try to minimize) how long it takes to run the valuation functionality, so long as it stays significantly below 50ms.
3. Contribute to architecture + loss function (apply some regularization, get a reasonable loss function, etc...) and run my first training run
4. Get a hyperparameter training run.

Target is 4PM.

Key agenda items afterwards are to:
- Try to add more inference time compute and see how much we can squeeze
- Investigate architecture + loss function improvements further
- Try to see how we could surmount the data wall (measuring stockfish latency, data preprocessing, etc...)
- Look into learning via self-play, RL, etc...
- Better tooling to test (i.e. locally or with a GUI)
- More board representations

Random ideas
- We CAN keep track of history
- We ONLY need to play for 50 turns per agent (100 in total)
- Stockfish rules as matrices
- Keep values bounded for targets