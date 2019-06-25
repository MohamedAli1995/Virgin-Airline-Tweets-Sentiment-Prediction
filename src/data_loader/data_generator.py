import numpy as np
from src.utils.utils import unpickle
from src.data_loader.preprocessing import preprocess_text, text_to_seq, one_hot_encoding
from random import shuffle
from glob import glob
import pandas as pd


class DataGenerator:
    """DataGenerator class responsible for dealing dataset with .

    Attributes:
        config: Config object to store data related to training, testing and validation.
        all_train_data: Contains the whole dataset(since the dataset fits in memory).
        x_all_train: Contains  the whole input training-data.
        x_all_train: Contains  the whole target_output labels for training-data.
        x_train: Contains training set inputs.
        y_train: Contains training set target output.
        x_val: Contains validation set inputs.
        y_val: Contains validation set target output.
    """

    def __init__(self, config, training=True, data_split_seed=64):
        self.config = config
        self.training = training
        self.data_split_seed = data_split_seed
        self.x_all_data = None
        self.y_all_data = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.all_test = None

        self.num_batches_train = None
        self.num_batches_val = None
        self.num_batches_test = None

        self.indx_batch_train = 0
        self.indx_batch_val = 0
        self.indx_batch_test = 0

        if self.training:
            np.random.seed(self.data_split_seed)
            self.__load_train_val_test(train_ratio=0.8, val_ratio=0.1)

    def __load_train_val_test(self, train_ratio=0.8, val_ratio=0.1):
        """Private function.
        Returns:
        """
        if train_ratio + val_ratio > 1.0:
            print("Wrong data splitting ratio.")
            return

        processed_df = preprocess_text(self.config.train_data_path)

        self.x_all_data = text_to_seq(processed_df['text'].values, create_tokenizer=True,
                                      max_features=self.config.max_n_words,
                                      tokenizer_pickle_path=self.config.tokenizer_pickle_path)

        self.y_all_data = one_hot_encoding(processed_df['airline_sentiment'].values, 3, dict=self.config.labels_dict)

        indices_list = [i for i in range(self.x_all_data.shape[0])]
        shuffle(indices_list)
        self.x_all_data = self.x_all_data[indices_list]
        self.y_all_data = self.y_all_data[indices_list]

        # Split train
        split_point_train = int(train_ratio * self.x_all_data.shape[0])
        self.x_train = self.x_all_data[0:split_point_train]
        self.y_train = self.y_all_data[0:split_point_train]
        # Split val
        split_point_val = int((val_ratio + train_ratio) * self.x_all_data.shape[0])
        self.x_val = self.x_all_data[split_point_train:split_point_val]
        self.y_val = self.y_all_data[split_point_train:split_point_val]
        # Split test
        self.x_test = self.x_all_data[split_point_val:self.x_all_data.shape[0]]
        self.y_test = self.y_all_data[split_point_val:self.x_all_data.shape[0]]

        self.num_batches_train = int(np.ceil(self.x_train.shape[0] / self.config.batch_size))
        self.num_batches_val = int(np.ceil(self.x_val.shape[0] / self.config.batch_size))
        self.num_batches_test = int(np.ceil(self.x_test.shape[0] / self.config.batch_size))

    def __shuffle_all_data(self):
        """Private function.
        Shuffles the whole training set to avoid patterns recognition by the model(I liked that course:D).
        shuffle function is used instead of sklearn shuffle function in order reduce usage of
        external dependencies.

        Returns:
        """
        indices_list = [i for i in range(self.x_train.shape[0])]
        shuffle(indices_list)
        # Next two lines may cause memory error if no sufficient ram.
        self.x_train = self.x_train[indices_list]
        self.y_train = self.y_train[indices_list]

        indices_list = [i for i in range(self.x_val.shape[0])]
        shuffle(indices_list)
        # Next two lines may cause memory error if no sufficient ram.
        self.x_val = self.x_val[indices_list]
        self.y_val = self.y_val[indices_list]

    def load_test_set(self, test_set):
        self.x_test = text_to_seq(test_set, create_tokenizer=False,
                                  tokenizer_pickle_path=self.config.tokenizer_pickle_path)
        self.all_test = self.x_test
        self.num_batches_test = int(np.ceil(self.x_test.shape[0] / self.config.batch_size))

    def prepare_new_epoch_data(self):
        """Prepares the dataset for a new epoch by setting the indx of the batches to 0 and shuffling
        the training data.

        Returns:
        """
        self.indx_batch_train = 0
        self.indx_batch_val = 0
        self.indx_batch_test = 0
        self.__shuffle_all_data()

    def next_batch(self, batch_type="train"):
        """Moves the indx_batch_... pointer to the next segment of the data.

        Args:
            batch_type: the type of the batch to be returned(train, test, validation, unlabeled_test).

        Returns:
            The next batch of the data with type of batch_type.
        """
        if batch_type == "unlabeled_test":
            x = self.x_test[self.indx_batch_test:self.indx_batch_test + self.config.batch_size]
            self.indx_batch_test = (self.indx_batch_test + self.config.batch_size) % self.x_test.shape[0]
            return x

        if batch_type == "train":
            x = self.x_train[self.indx_batch_train:self.indx_batch_train + self.config.batch_size]
            y = self.y_train[self.indx_batch_train:self.indx_batch_train + self.config.batch_size]
            self.indx_batch_train = (self.indx_batch_train + self.config.batch_size) % self.x_train.shape[0]

        elif batch_type == "val":
            x = self.x_val[self.indx_batch_val:self.indx_batch_val + self.config.batch_size]
            y = self.y_val[self.indx_batch_val:self.indx_batch_val + self.config.batch_size]
            self.indx_batch_val = (self.indx_batch_val + self.config.batch_size) % self.x_val.shape[0]

        elif batch_type == "test":
            x = self.x_test[self.indx_batch_test:self.indx_batch_test + self.config.batch_size]
            y = self.y_test[self.indx_batch_test:self.indx_batch_test + self.config.batch_size]
            self.indx_batch_test = (self.indx_batch_test + self.config.batch_size) % self.x_test.shape[0]

        return x, y
