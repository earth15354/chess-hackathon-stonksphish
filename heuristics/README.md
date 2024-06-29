# Heuristics
The easiest way to get performance akin to stockfish, to begin with, will be to use a bunch of heuristics based on known good and bad properties of board states.

Heuristics V1 is mainly by Claude and is based on VALUES for the board. Heuristics V2 is going to be a more generalized heuristics function based on four types (conceptually) of heuristics:
1. Piece-wise valuation, independent of other pieces: will create table(s)
2. Position-wise valuation, independent of the pieces: will create table(s)
3. Position-wise valuation taking into account pieces
4. Boolean aggregates of the entire space

We stack the first three into one large mega-table along with the incoming tensor, and the last one becomes a secondary input. We try to train so that the representations will be aligned between these two and then CNN/Attention on top will reduce it to moves/valuations.