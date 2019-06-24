import _pickle as cPickle
import argparse as arg
import os


def get_args():
    argparse = arg.ArgumentParser(description=__doc__)

    argparse.add_argument(
        '-c', '--config',
        metavar='c',
        help='Config file path')

    argparse.add_argument(
        '-i', '--input_text',
        metavar='i',
        help='input text')

    args = argparse.parse_args()
    return args


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def print_predictions( predictions):
    labels = ["positive, neutral, negative"]
    print("Predictions:\n")
    for i in range(predictions.shape[0]):

        print("Prediction:%s\n" % (labels[int(predictions[i])]))
