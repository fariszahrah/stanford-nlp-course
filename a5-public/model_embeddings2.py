
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h


        self.embed_size = word_embed_size
        self.vocab = vocab
        self.char_embed_size = 50
        self.max_word_len = 21
        self.dropout_rate = 0.2

        self.char_embedding  = nn.Embedding(
            num_embeddings =len(vocab.char2id),
            embedding_dim  =self.char_embed_size,
            padding_idx    =vocab.char2id['<pad>'],

        )
        self.CNN = CNN(embed_size = self.char_embed_size,
                        out_channels=self.embed_size,
                        word_len =self.max_word_len)
        self.Highway = Highway(self.embed_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """

        print('input.size: ', input.size())
        char_embeddings = self.char_embedding(input) # sentence_length, batch_size, max_word_length,
        sent_len, batch_size, max_word, _ = char_embeddings.shape
        view_shape = (sent_len * batch_size, max_word, self.char_embed_size)
        # bb = sent_len * batch_size
        # bb, char_embed, max_word because 1D CNN only convolve in last dimension
        char_embeddings = char_embeddings.view(view_shape).transpose(1, 2)
        print('Char embedding size: ',char_embeddings.size())

        x_conv    = self.CNN(char_embeddings) # bb, word_embed_size
        x_highway = self.Highway(x_conv)
        output    = self.dropout(x_highway) # bb, word_embed
        output    = output.view(sent_len, batch_size, self.embed_size)
        return output
