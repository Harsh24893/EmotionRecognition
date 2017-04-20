### Project: Emotion Detection from Text

### Highlights:

 - This is a **multi-class text classification (sentence classification)** problem.
 - The purpose of this project is to **classify emotions from sentences into 7 different categories**.
 - The model was built with **Convolutional Neural Network (CNN)** and **Word Embeddings** on **Tensorflow**.

### Train:

 - To run CNN for classifcation -- Command: python train_cnn.py
 - To run LSTM for classification -- Command: python lstm.py 
 
 A directory will be created during training, and the best model will be saved in this directory. 


### Reference:
 - [Implement a cnn for text classification in tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
 - [UNDERSTANDING CONVOLUTIONAL NEURAL NETWORKS FOR NLP] (http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
 - [Understanding LSTM for text classification] (https://github.com/hx364/NN_sentiment)
 
 
### In order to run this download Word2Vec and Glove Vectors, since the file sizes are very large.
 - [Word2Vec] (https://nlp.stanford.edu/projects/glove/)
 - [Glove] (https://radimrehurek.com/gensim/models/word2vec.html)