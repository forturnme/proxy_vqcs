# Proxy VQCS
Search for VQC with proxy model.
Implemented on `pymoo`, `scikit-learn`, `mindspore` and `mindquantum`, for VQE problems.

# Usage
First, get samples from `sample.py`. Recommended sample circuits is 100 for the test case (H4 linear molecule).

Second, fit the sampled data into a random forest regressor with 50 trees with `fit_rf.py`.

Then, run `train_proxy.py` to train and save a `mindspore` MLP with the sampled circuits and random circuits with predicted loss by random forest. The portion is about 24:8 for every epoch. Here it is trained for 50 epoches.

Finally, search for a generation of circuits (which have 20 circuits in this case) using the trained MLP as evaluator by running `search.py`, and yield the best circuit from final population.

## Other scripts:

`evaluator.py` contains the evaluator function for VQE which takes a circuit code as input, and returns its loss value.

`search_no_proxy.py` search for circuits, which directly measure its circuits on real evaluator but not proxy models.

`search_rf.py` search for circuits, which measure its circuits on the fitted random forest model.

`circuit_codec.py` contains the translator from discrete code to circuit, from continuous code to discrete code and from continuous code to circuit.
