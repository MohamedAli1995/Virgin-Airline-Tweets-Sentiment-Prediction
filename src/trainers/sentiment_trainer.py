from src.base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class SentimentTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(SentimentTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.data.num_batches_train))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)
        print("EPOCH: [", self.model.cur_epoch_tensor.eval(self.sess), "] train_accuracy: ",
              acc * 100, "% train_loss: ", loss)
        if loss < 0:
            print(" error in loss")
            exit()
        cur_it = self.model.global_step_tensor.eval(self.sess)

        summaries_dict = {
            'loss': loss,
            'acc': acc
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = self.data.next_batch(batch_type="train")
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True,
                     self.model.seq_len: batch_x.shape[1], self.model.keep_prob_lstm_out: 0.2,
                     self.model.keep_prob_lstm_recurrent: 0.2, self.model.keep_prob_fc: 0.5}

        # Run training for step first with dropout then calculate loss and acc without dropout.
        self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                      feed_dict=feed_dict)
        # Run without dropout
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: False,
                     self.model.seq_len: batch_x.shape[1],
                     self.model.keep_prob_lstm_out: 1.0,
                     self.model.keep_prob_lstm_recurrent: 1.0, self.model.keep_prob_fc: 1.0}
        loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy],
                                  feed_dict=feed_dict)
        return loss, acc

    def validate_epoch(self):
        loop = tqdm(range(self.data.num_batches_val))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.validate_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)
        print("EPOCH: [", self.model.cur_epoch_tensor.eval(self.sess), "] val_accuracy: ",
              acc * 100, "% val_loss: ", loss)

        summaries_dict = {
            'val_loss': loss,
            'val_acc': acc
        }
        cur_it = self.model.global_step_tensor.eval(self.sess)
        self.logger.summarize(cur_it, scope="val", summaries_dict=summaries_dict)

    def validate_step(self):
        batch_x, batch_y = self.data.next_batch(batch_type="val")
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: False,
                     self.model.seq_len: batch_x.shape[1],
                     self.model.keep_prob_lstm_out: 1.0,
                     self.model.keep_prob_lstm_recurrent: 1.0, self.model.keep_prob_fc: 1.0}

        loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy],
                                  feed_dict=feed_dict)

        return loss, acc
