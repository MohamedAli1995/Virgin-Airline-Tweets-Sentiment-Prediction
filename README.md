# VirginAirline-tweets-sentiment-prediction
Sentiment analysis network that predicts the opinion ( positive, neutral and negative) about a given text.
 
This project follows the **best practice tensorflow folder structure of** [Tensorflow Best Practice](https://github.com/MrGemy95/Tensorflow-Project-Template) 


# Table of contents

- [Project structure](#project-structure)
- [Download pretrained models](#Download-pretrained-models)
- [Dependencies](#install-dependencies)
- [Config file](#config-file)
- [How to train](#Model-training)
- [How to test](#Model-testing)
- [How to predict class of images using pretrained models](#Make-predictions-with-pretrained-models)
- [Implementation details](#Implementation-details)
     - [TinyVGG architecture](#TinyVGG-model-arch)
     - [TinyVGG training](#TinyVGG-training)
     - [TinyVGG testing](#TinyVGG-testing)



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
├── models              - This folder contains 2 models implemented for cifar-100.
│   └── sentiment_model.py  - Contains the architecture of the Sentiment(LSTM) model used
│
│
├── trainer             - This folder contains trainers used which inherit from BaseTrain.
│   └── sentiment_trainer.py - Contains the trainer class of the Sentiment model.
│ 
|
├── testers             - This folder contains testers used which inherit from BaseTest.
│   └── sentiment_tester.py - Contains the tester class of the TinyVGG model.
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

# Model training
In order to train, pretrain or test the model you need first to edit the config file that is described at(#Config-File).<br>
To train a Sentiment LSTM model:<br>
set:<br>
```
"num_epochs":200,
"learning_rate":0.0001,
"batch_size":256,

"train_data_path": set it to path of the training data e.g: "/content/train"
"meta_data_path": path to metadata of the training set, e.g: "/content/cifar-100-python/meta"
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

## Sentiment model arch
<img src="https://github.com/MohamedAli1995/Virgin-Airline-Tweets-Sentiment-Prediction/blob/master/diagrams/model_diagram.png"
     alt="Image not loaded"
     style="float: left; margin-right: 10px;" />

## model training
 I trained the Sentiment  model by splitting training_data into train/val/test with ratios 8:1:1 for 200 epoch<br>
 Acheived val accuracy of 80<br>
 and training accuracy of 50% (with enabling dropout)<br>
 <img src="https://github.com/MohamedAli1995/Cifar-100-Classifier/blob/master/src/models/train_accuracy.png"
     alt="Image not loaded"
     style="float: left; margin-right: 10px;" />
and loss <br>
 <img src="https://github.com/MohamedAli1995/Cifar-100-Classifier/blob/master/src/models/loss.png"
     alt="Image not loaded"
     style="float: left; margin-right: 10px;" />
     
## model testing
   Acheived testing accuracy of 52% on the cifar-100 test set 


