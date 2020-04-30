## Chatbot Implementation on tensorflow 1.0.0

This is a sample implementation of a state-of-the-art chatbot Seq-Seq model, to be trained on the cornell movie
dialog dataset. The dataset can also be taken from more sources like the twitter and reddit dataset,
which are readyly available on the internet. The chatbot is based on Deep NLP, having RNN network for both 
encoder and the decoder implementation. The implementation is given in the chatbot.py file in the repo.
The questions and target are preprocessed and mapped to integer labels and is propagated through the encoder and 
decoder network, forward and back. 

A better and robust implementation of a Seq2Seq model is given in the following repo :
https://github.com/AbrahamSanders/seq2seq-chatbot

The above repo has the trained model on tensorflow 0.12.0, and the build instruction. 
We have the weights checkpoint in the trained_v_modelv1 folder, that can we used for 
PCA(Principal Component Analysis) for visualizing the data and the weights, and the nearest 
neightbours for words on Tensorboard. 
The chatbot also works brilliantly.

Other implementation of such models on tensorflow 1.4.0 and PyTorch is attached along with the repo.
Disclaimer:These are not my contributions.
Other documentations on more chatbot models : 
http://complx.me/2016-06-28-easy-seq2seq/
http://complx.me/2016-12-31-practical-seq2seq/

