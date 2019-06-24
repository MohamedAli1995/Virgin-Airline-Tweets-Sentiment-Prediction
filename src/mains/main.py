import tensorflow as tf
from src.data_loader.data_generator import DataGenerator
from src.models.gesture_recognition_model import GestureRecognitionModel
from src.trainers.gesture_recognition_trainer import GestureRecognitionTrainer
from src.testers.gesture_recognition_tester import GestureRecognitionTester
from src.utils.config import processing_config
from src.utils.logger import Logger
from src.utils.utils import get_args
from src.utils.utils import print_predictions


def main():
    args = None
    config = None
    try:
        args = get_args()

        config = processing_config("/media/syrix/programms/projects/Virgin-Airline-Tweets-Sentiment-Prediction/configs/config_model.json")
        # config = processing_config(
        #     "/media/syrix/programms/projects/Virgin-Airline-Tweets-Sentiment-Prediction/configs/config_model.json")
    except:
        print("Missing or invalid arguments")
        exit(0)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = config.per_process_gpu_memory_fraction
    sess = tf.Session()
    logger = Logger(sess, config)
    model = GestureRecognitionModel(config)
    model.load(sess)

    if args.input_text is not None:
        data = DataGenerator(config, training=False)
        data.load_test_set(args.input_text)
        tester = GestureRecognitionTester(sess, model, data, config, logger)
        predictions = tester.predict()
        print_predictions(predictions)
        return



    data = DataGenerator(config, training=True)

    trainer = GestureRecognitionTrainer(sess, model, data, config, logger)
    trainer.train()
    tester = GestureRecognitionTester(sess, model, data, config, logger)
    tester.test()


if __name__ == '__main__':
    main()
