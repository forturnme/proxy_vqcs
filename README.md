# Proxy VQCS
Search for VQC with proxy model.
Implemented on `pymoo`, `scikit-learn`, `mindspore` and `mindquantum`, for VQE problems.

For test purpose, just run `bash test_all.sh`

# Usage
Before all options, pretrain a set of params and save them. Use `pretrain.py`.

First, get samples from `sample.py`. Recommended sample circuits is 100 for the test case (H4 linear molecule).

Second, fit the sampled data into a random forest regressor with 50 trees with `fit_rf.py`.

Then, run `train_proxy.py` to train and save a `mindspore` MLP with the sampled circuits and random circuits with predicted loss by random forest. The portion is about 24:8 for every epoch. Here it is trained for 50 epoches.

Finally, search for a generation of circuits (which have 20 circuits in this case) using the trained MLP as evaluator by running `search.py`, and yield the best circuit from final population.

## Other scripts:
`params.py` defines the class to yield, hold, save and load supernet params in its matrix.

`evaluator.py` contains the evaluator function for VQE which takes a circuit code as input, and returns its loss value.

`search_no_proxy.py` search for circuits, which directly measure its circuits on real evaluator but not proxy models.

`search_rf.py` search for circuits, which measure its circuits on the fitted random forest model.

`codec.py` contains the translator from discrete code to circuit, from continuous code to discrete code and from continuous code to circuit. Of course, the circuit blocks can be repeated, but it was not implemented.

`circuit_module.py` defines the single qubit rotator and the entagler gate.

## Build iteratively
You can build the first part of the circuit by this algorithm, and then store the circuit and parameters, and then build the next part on the top of it. This is a achievable polish direction of this work.
