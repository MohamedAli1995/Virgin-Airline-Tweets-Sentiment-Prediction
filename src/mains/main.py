import tensorflow as tf
from src.data_loader.data_generator import DataGenerator
from src.models.sentiment_model import SentimentModel
from src.trainers.sentiment_trainer import SentimentTrainer
from src.testers.sentiment_tester import SentimentTester
from src.utils.config import processing_config
from src.utils.logger import Logger
from src.utils.utils import get_args

from src.utils.utils import print_predictions


def main():
    args = None
    config = None
    try:
        args = get_args()
        config = processing_config(args.config)
        # config = processing_config(
        #     "/media/syrix/programms/projects/Virgin-Airline-Tweets-Sentiment-Prediction/configs/config_model.json")
    except:
        print("Missing or invalid arguments")
        exit(0)

    sess = tf.Session()
    logger = Logger(sess, config)
    model = SentimentModel(config)
    model.load(sess)

    if args.input_text is not None:
        data = DataGenerator(config, training=False)
        data.load_test_set([args.input_text])
        tester = SentimentTester(sess, model, data, config, logger)
        predictions = tester.predict()
        print_predictions(predictions)
        return

    data = DataGenerator(config, training=True)

    trainer = SentimentTrainer(sess, model, data, config, logger)
    trainer.train()
    tester = SentimentTester(sess, model, data, config, logger)
    tester.test()


if __name__ == '__main__':
    main()
