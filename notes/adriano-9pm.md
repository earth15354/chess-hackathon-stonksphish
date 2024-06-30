Seems like it would be smart to have a bunch of multimodal and contrastive learning being trained on the same time. We could try and force the learned representation to both be usable for a bunch of tasks that we think are useful for evaluation of a game state, including not only the value but also what the likely next moves are, and more.

Maybe we can use some sort of search procedure over a large neural memory bank to try and match existing games that have been seen?

Maybe we could massively train on seeing the previous move instead.