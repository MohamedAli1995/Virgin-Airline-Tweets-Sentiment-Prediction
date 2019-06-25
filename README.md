# Virgin-Airline-tweets-sentiment-prediction
Sentiment analysis network that predicts the opinion ( positive, neutral and negative) about a given text.<br>
### Quick Example
```
1. input = "the flight was good" => output: positive
2. input = "the flight was bad" => output: negative
3. input = "hello world" => output: neutral
```
This project structure follows the **best practice tensorflow folder structure of** [Tensorflow Best Practice](https://github.com/MrGemy95/Tensorflow-Project-Template) 


# Table of contents

- [Project structure](#project-structure)
- [Download pretrained models](#Download-pretrained-models)
- [Dependencies](#install-dependencies)
- [Config file](#config-file)
- [How to train](#How-to-Train)
- [How to predict](#Make-predictions-with-pretrained-models)
- [Implementation details](#Implementation-details)
     - [Preprocessing](#Sentiment-model-preprocessing)
          - [Ineresting columns](#Interesting-columns)
          - [Lower case](#Lower-case)
          - [Don't remove stop words](#Don't-remove-stop-words)
          - [Removing symbols](#Removing-symbols)
          - [Text2seq](#Tokenization)
          - [Padding](#Padding)
          - [Saving tokenizer](#Saving-tokenizer)
          - [One hot encoding labels](#One-hot-encoding-labels)
          - [Dataset Shuffling](#Shuffling-dataset)
          - [Splitting train, val and test](#Splitting-dataset)
     - [Sentiment model architecture](#Sentiment-model-arch)
     - [Model training](#Model-Training)


# Project structure
--------------

```
├── Configs
│   └── config_model.json  - Contains the paths used and config of the models(learning_rate, num_epochs, ...)
│ 
├──  base
│   ├── base_model.py   - This file contains the abstract class of all models used.
│   ├── base_train.py   - This file contains the abstract class of the trainer of all models used.
│   └── base_test.py    - This file contains the abstract class of the testers of all models used.
│
├── models              - This folder contains 1 model for sentiment analysis.
│   └── sentiment_model.py  - Contains the architecture of the Sentiment(LSTM) model used
│
│
├── trainer             - This folder contains trainers used which inherit from BaseTrain.
│   └── sentiment_trainer.py - Contains the trainer class of the Sentiment model.
│ 
|
├── testers             - This folder contains testers used which inherit from BaseTest.
│   └── sentiment_tester.py - Contains the tester class of the Sentiment model model.
│ 
| 
├──  mains 
│    └── main.py  - responsible for the whole pipeline.
|
│ 
├──  data _loader 
│    ├── data_generator.py  - Contains DataGenerator class which handles Virgin Airline Tweets dataset.
│    └── preprocessing.py   - Contains helper functions for preprocessing Virgin Airline Tweets dataset.
| 
└── utils
     ├── config.py  - Contains utility functions to handle json config file.
     ├── logger.py  - Contains Logger class which handles tensorboard.
     └── utils.py   - Contains utility functions to parse arguments and handle pickle data. 
```


# Download pretrained models:
Pretrained models can be found at saved_models/checkpoint

# Install dependencies

* Python3.x <br>

* [Tensorflow](https://www.tensorflow.org/install)

* Tensorboard[optional] <br>

* Numpy
```
pip3 install numpy
```

* Bunch
```
pip3 install bunch
```

* Pandas
```
pip3 install pandas
```

* tqdm
```
pip3 install tqdm
```

# Config File
In order to train, pretrain or test the model you need first to edit the config file:
```
{
  "num_epochs": 200,               - Numer of epochs to train the model if it is in train mode.
  "learning_rate": 0.001,         - Learning rate used for training the model.
  "batch_size": 256,               - Batch size for training, validation and testing sets(#TODO: edit single batch_size per mode)
  "val_per_epoch": 1,              - Get validation set acc and loss per val_per_epoch. (Can be ignored).
  "max_to_keep":1,                 - Maximum number of checkpoints to keep.

  "train_data_path":"path_to_training_set",                      - Path to training data.
  "test_data_path":"path_to_test_set",                           - Path to test data.
  "checkpoint_dir":"path_to_store_the_model_checkpoints",        - Path to checkpoints store location/ or loading model.
  "summary_dir":"path_to_store_model_summaries_for_tensorboard",  - Path to summaries store location/.
  "tokenizer_pickle_path":"", - Path to tokenizer pickle file saved, it is used for text_to_sequence

}
```

# How to Train
In order to train, pretrain or test the model you need first to edit the config file that is described at[config file](#config-file).<br>
To train a Sentiment LSTM model:<br>
set:<br>
```
"num_epochs":200,
"learning_rate":0.0001,
"batch_size":256,

"tokenizer_pickle_path":"", - Path to tokenizer pickle file saved, it is used for text_to_sequence
"train_data_path": set it to path of the training data e.g: "/content/train"
"checkpoint_dir": path to store checkpoints, e.g: "/content/saved_models/tiny_vgg_model/checkpoint/"
"summary_dir": path to store the model summaries for tensorboard, e.g: "/content/saved_models/tiny_vgg_model/summary/"
```
Then change directory to the project's folder and run:
```
python3.6 -m src.mains.main --config path_to_config_file
```
# Make predictions with pretrained models
To make predictions using text input:<br>
```
python3.6 -m src.mains.main --config path_to_config_file -i "text to analyze"
```
# Implementation details
## Sentiment model preprocessing
talk about preprocessing
### Interesting columns
After dataset exploration, we find out that we have only two columns that we are interested in, "text" and "airline_sentiment", the first is our input and the last is our target output <br>
```
df = pd.read_csv(path)
# Select only text & airline_sentiment fields.
df = df[["text", "airline_sentiment"]]
```
### Lower case
First we convert the dataset to lowercase, since the context is case independent(unless you write UPPER CASE when you are angry, we will ignore this for now:D) <br>
```
df['text'] = df['text'].apply(lambda x: x.lower())
```
### Don't remove stop words
Even if stop words is incredibly frequent, removin stop words can affect the context, we won't remove it <br>
### Removing symbols
As we are analyzing tweets, we have a lot of symbols to remove, more important, we should eliminate words starts with @, for example @mohamed_ali should be eliminated not only the symbol @<br>
```
# Remove any symbols except @.
df['text'] = df['text'].apply((lambda x: re.sub('[^@a-zA-z\s]', '', x)))
# Remove anyword having @ in it, as it is a tag operator.
df['text'] = df['text'].apply(lambda x: re.sub('[\w]*[@+][\w]*[\s]*', '', x))
```
### Tokenization
After preprocessing the text, we need to convert it to numbers in order to feed it to our embedding layer to convert it to a dense vector representation<br>
```
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(text_tuples)
sequences = tokenizer.texts_to_sequences(text_tuples)
```
### Padding
Then we pad our sequences with zeros to the max length, a drawback here is that our model doesn't accept dynamic sequence length, we will figure out how to solve such a problem later<br>
```
x = pad_sequences(x, maxlen=30)
```
### Saving tokenizer
Finally we save our tokenizer as a pickle file, in order to use it when testing input, as we will not have our dataset then<br>
```
pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
```
### One hot encoding labels
We one-hot encode our labels{positive, neutral, negative} to a sparse vector representation<br>
positive => [1, 0, 0]<br>
neutral =>[0, 1, 0] <br>
negative => [0, 0, 1]<br>

### Shuffling dataset
After we have preprocessed our dataset, we first shuffle our dataset once before splitting, thenwe shuffle our training set every epoch in order to decrease overfitting and increase learning curve <br>
```
indices_list = [i for i in range(self.x_all_data.shape[0])]
shuffle(indices_list)
self.x_all_data = self.x_all_data[indices_list]
self.y_all_data = self.y_all_data[indices_list]
```
### Splitting dataset
We split dataset into training, validation and test sets with ratio (8:1:1)<br>
          
## Sentiment model arch
<img src="https://github.com/MohamedAli1995/Virgin-Airline-Tweets-Sentiment-Prediction/blob/master/saved_models/diagrams/gesture_recognition_model_arch.png"  height="50%"
     alt="Image not loaded" style="float: left; margin-right: 10px;" />

## Model Training
 I trained the Sentiment  model by splitting training_data into train/val/test with ratios 8:1:1 for 200 epoch<br>
 Acheived val accuracy of 88%, val_loss of 0.2872<br>
 training accuracy of 90%, training_loss of 0.26895<br><br>

model val_acc <br>
<img src="https://github.com/MohamedAli1995/Virgin-Airline-Tweets-Sentiment-Prediction/blob/master/saved_models/diagrams/val_acc.png" alt="Image not loaded" style="float: left; margin-right: 10px;" />

     alt="Image not loaded"
     style="float: left; margin-right: 10px;" />

and loss <br>
<img src="https://github.com/MohamedAli1995/Virgin-Airline-Tweets-Sentiment-Prediction/blob/master/saved_models/diagrams/val_loss.png"
     alt="Image not loaded"
     style="float: left; margin-right: 10px;" />
     
## model testing
   Acheived testing accuracy of 88% on 10% of the dataset (unseen in training process).<br>
   with test loss of 0.267
