import logging
import tensorflow as tf
from fedml import mlops
from fedml.core import ServerAggregator
import os

class TfServerAggregator(ServerAggregator):
    def __init__(self, model, dataset, args):
        super().__init__(model, args)
        self.model = model
        self.test_img_datagen = dataset[3]
        self.test_dataset_length = dataset[1]
        logging.error("server aggregator - constructor initailized")

    def get_model_params(self):
        logging.error("server aggregator - get_model_params")
        return self.model.get_weights()

    def set_model_params(self, model_parameters):
        logging.error("server aggregator - set_model_params")
        self.model.set_weights(model_parameters)

    def test(self, test_data, device, args):
        logging.error("server aggregator - Evaluating on Trainer ID: {}".format(self.id))
        loss, dice_coef, soft_dice_coef = self.model.evaluate(self.test_img_datagen, steps=self.test_dataset_length//args.batch_size)
        metrics = {"loss": loss, 
                   "dice_coef": dice_coef, 
                   "soft_dice_coef": soft_dice_coef
                   }
        logging.error("server aggregator - loss", loss)
        logging.error("server aggregator - dice_coef", dice_coef)
        logging.error("server aggregator - soft_dice_coef", soft_dice_coef)
        self.model.save(os.path.join("", "content", "Aggregated_model.h5"))
        return metrics

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        loss, dice_coef, soft_dice_coef = self.model.evaluate(self.test_img_datagen, steps=self.test_dataset_length//args.batch_size)
        metrics = {"loss": loss, 
                   "dice_coef": dice_coef, 
                   "soft_dice_coef": soft_dice_coef
                   }
        logging.error("server aggregator - loss", loss)
        logging.error("server aggregator - dice_coef", dice_coef)
        logging.error("server aggregator - soft_dice_coef", soft_dice_coef)
        self.model.save(os.path.join("", "content", "Aggregated_model.h5"))
        return True