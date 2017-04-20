### Project: Emotion Detection from Text

###Team :
 - [Navnith](https://github.com/navnith)
 - [Pragya](https://github.com/pragyaarora)
 - [Piyush](https://github.com/piyushghai)
 - [Yours Truly](https://github.com/Harsh24893)

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
 - [UNDERSTANDING CONVOLUTIONAL NEURAL NETWORKS FOR NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
 - [Understanding LSTM for text classification](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
 
 
### In order to run this download Word2Vec and Glove Vectors, since the file sizes are very large.
 - [Glove] (https://nlp.stanford.edu/projects/glove/)
 - [Word2Vec] (https://radimrehurek.com/gensim/models/word2vec.html)