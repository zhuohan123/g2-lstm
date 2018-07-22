# g2-lstm
Codes for "Towards Binary-Valued Gates for Robust LSTM Training".

Language modeling code is based on [awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm) using PyTorch.

Translation code is based on Theano.

Implementation of Gumbel-Gate LSTM: [Pytorch version](language-modeling/g2_lstm.py), [Theano version](machine-translation/libs/layers/stochastic_lstm.py).

We also apply *dropout* to the Gumbel noise added to the gates. In particular, given a fixed probability *p*, all gates will independently be preturbed by the Gumbel noise with probability *p*, or stay unperturbed otherwise. We find that no matter what the value of *p* is, the performance of trained G2-LSTM will be better. When *p* is small, our model will have better generalization error, and when *p* is large, our model will have less performance drop under compression. We fix *p=0.2* in all our experiments in the paper.
