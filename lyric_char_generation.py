
# coding: utf-8

# In[1]:


import pandas as pd
import os
import string
import re
import numpy as np


#!git clone https://github.com/davordavidovic/NLP-lyrics-generator.git
  
#!sudo pip install h5py


# In[46]:


def load_songs(genre, max_tokens):
    df1 = pd.read_csv('data/lyrics_part1.csv')
    df2 = pd.read_csv('data/lyrics_part2.csv')
    df3 = pd.read_csv('data/lyrics_part3.csv')
    df4 = pd.read_csv('data/lyrics_part4.csv')

    df_part_1 = pd.concat([df1, df2])
    df_part_2 = pd.concat([df3, df4])

    df = pd.concat([df_part_1, df_part_2])
    df.drop(columns=['index','Unnamed: 0'], inplace=True) #we dont need these columns

    df = df.dropna() #there were around 10000 rows with no lyrics so drop them

    df_songs = df[df.genre==genre]

    df_songs['preprocessed'] = df_songs['lyrics'].map(prepare_text)

    songs = df_songs.preprocessed.values
    
    count = 0
    cut = 0
    for i,song in enumerate(songs):
        tokens = list(song)
        count += len(tokens) 
        if count >= max_tokens:
            cut = i - 1
            break
    
    return songs[:cut]


# In[3]:


def prepare_text(text):
    text = text.lower()
    text = text.replace('\n', ' N ')
  
    text = text.split()

    for index, word in enumerate(text):
        #remove non alphabetic characters at the end or beginning of a word
        word = word.strip(string.punctuation)
    
        #replace non alhpanumeric chars with space
        word = re.sub(r"[\W]",' ',word)
        text[index] = word 
   
    #concatenate again
    text = " ".join(text)
    return text


# In[4]:


def build_vocab(songs):
    # create mapping of unique chars to integers
    chars = sorted(list(set(" ".join(songs))))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    return chars, char_to_int


# In[5]:


def index2sen(seq,chars):
    tokens = [chars[int(t)] for t in seq]
    sen = "".join(tokens)
    return sen


# In[43]:


import numpy as np

def songs_to_supervised(seq_len, songs, char_to_int):
    data_x = []
    data_y = []

    for song in songs:
        tokens = list(song)
        for i in range(0, len(tokens) - seq_len):
            seq_in = tokens[i:i+seq_len]
            seq_out = tokens[i + seq_len]
            seq_data = []
            for c in seq_in:
                data_x.append(char_to_int[c])

    return data_x, data_y


# In[9]:



from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense,Dropout, CuDNNLSTM
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
import keras.utils as ku 

def create_model(layers, units, inp_shape, out_shape):
    #lstm sequence to categoriemodel
    model = Sequential()
  
    for l in range(layers-1):
        model.add(CuDNNLSTM(units,return_sequences=True, input_shape = inp_shape))
        model.add(Dropout(0.2))
    
    model.add(CuDNNLSTM(units,return_sequences=False))
    model.add(Dropout(0.2)) 
    model.add(Dense(out_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
  
    return model


# In[10]:


def generate_text(seed_text, next_chars, model, chars, chars_to_int):
    seq_in = list(prepare_text(seed_text))
    x = np.array([chars_to_int[c] for c in seq_in])
    
    predictions = []
    for i in range(next_words):
        input_seq = np.reshape(np.append(x[i:],predictions),(1,len(x),1))
        predicted = model.predict_classes(input_seq, verbose=0)
        predictions.append(predicted[0])
        output_word = chars[predicted[0]]
        seed_text += " " + output_word
        
    return seed_text



# In[44]:


from keras.utils import np_utils
import numpy as np 
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle
 

def run_experiment(n_sequences, n_epochs, genre, seq_len, n_layers, directory):

    print("Running", n_sequences,"sequences", n_epochs,"epochs",genre, seq_len,"sequence length", n_layers, "layers", "vocab size", directory, "directory") 

    #load lyrics with this many tokens
    max_tokens = n_sequences-seq_len

    #load song lyrics
    songs = load_songs(genre, max_tokens)

    #create the vocabulary from the songs 
    chars, chars_to_int = build_vocab(songs)
    n_vocab = len(chars)
    #songs to sequences and labels
    data_x, data_y = songs_to_supervised(seq_len, songs, chars_to_int)
    
    
    #reshape input to samples, timesteps, features
    X = np.reshape(data_x, (len(data_x), seq_len, 1))
    #normalize input
    X = X/float(n_vocab)
    #categorical labels 
    y = np_utils.to_categorical(data_y)

    inp_shape = X[0].shape
    out_shape = y[0].shape[0]
    print("X shape",X.shape)
    #create the lstm model
    model = create_model(n_layers, units=400, inp_shape =inp_shape, out_shape=out_shape)

    # checkpoint
    #TODO adapt filepath
    filepath = directory + "weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='min')

    #early stopping 
    es = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=100)

    callbacks_list = [es]

    #train model
    history = model.fit(X, y, epochs=n_epochs, verbose=1,batch_size=1024,callbacks=callbacks_list, validation_split=0.1)

    #save model TODO namin
    model.save(directory +"model.h5")

    #save history
    with open(directory+"hist", 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    #generate validation texts and training texts
    val_words = seq_words[:-10]
    for t in val_words:
        sentence = " ".join(t[0])
        label = t[1]
        output = generate_text(sentence, next_words = seq_len, model = model, vocab_list = vocab_list)
        with open(directory + "generated.txt","w") as file:
            file.write(sentence + " out: " + output + "\n")
            #also save the actual number of sequences that were used
            file.write(str(len(data_x)))
  
  
    #TODO save plot on training curve
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'test'], loc='upper left')
    plot_path = directory + "plot.png"
    plt.savefig(plot_path, bbox_inches='tight', format='png')


# In[ ]:


data_sizes = [10000, 100000, 5000000] #num of sequences
epochs = [50,200]
genres = ['Pop', 'Hip-Hop', 'Metal', 'Country']
seq_lens = [20,50]
layers = [4, 8] #400 units each

experiments = []

#big dataset on all genres with different vocabulary sizes
for g in genres:
    exp = {"seqs" : data_sizes[2],
           "epochs" : epochs[1],
           "genre" : g,
           "seq_lens" : seq_lens[1],
           "layers" : layers[0],
           "dir" : "exps2/_" +  g
          }
    experiments.append(exp)

print("Running", len(experiments), "experiments")
            
for e in experiments:
    #try:
        n_seqs = e["seqs"]
        n_epochs = e["epochs"]
        genre = e["genre"]
        seq_len = e["seq_lens"]
        n_layers = e["layers"]
        dir_ = e["dir"]
        run_experiment(n_sequences = n_seqs, n_epochs = n_epochs, genre = genre, seq_len = seq_len, n_layers = n_layers, directory = dir_)
    #except Exception as ex:
     #   print(ex)
      #  pass

