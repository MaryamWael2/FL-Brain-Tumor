import logging
from fedml.core import ClientTrainer
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os

class TfModelTrainerCLS(ClientTrainer):
    def __init__(self, model, dataset, args):
        super().__init__(model, args)
        self.model = model
        self.train_img_datagen = dataset[2]
        # self.val_img_datagen = dataset[4]
        self.train_dataset_length = dataset[0]
        # self.val_dataset_length = dataset[1]
        logging.error("client trainer - constructor called")
        
    def get_model_params(self):
        logging.error("client trainer - get_model_params")
        return self.model.get_weights()

    def set_model_params(self, model_parameters):
        logging.error("client trainer - set_model_params")
        self.model.set_weights(model_parameters)

    def train(self, train_data, device, args):
        logging.error("client trainer - Start training on Trainer {}".format(self.id))
        batch_size: int = args.batch_size
        epochs: int = args.epochs
        
        filepath= os.path.join(".", 'client_model.h5')
        checkpoint = ModelCheckpoint(filepath, 
                                     save_best_only=True, monitor='loss',
                                     verbose=1, mode='min', save_freq="epoch")
        
        steps_per_epoch = self.train_dataset_length // batch_size
        # validation_steps = self.val_dataset_length // batch_size
        
        logging.error("client trainer - here before training")
        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.train_img_datagen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            # validation_data= self.val_img_datagen,
            # validation_steps = validation_steps,
            callbacks=checkpoint,
            verbose=1)
        logging.error("client trainer -here after training")

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        results = {
            "loss": history.history["loss"][0],
            "dice_coef": history.history["dice_coef"][0],
            # "val_loss": history.history["val_loss"][0],
            "soft_dice_coef": history.history["soft_dice_coef"][0],
        }
        
        logging.error("client trainer - metrics - loss", history.history["loss"][0])
        logging.error("client trainer - metrics - dice_coef", history.history["dice_coef"][0])
        logging.error("client trainer - metrics - soft_dice_coef", history.history["soft_dice_coef"][0])
        logging.error(
                "client trainer - Client Index = {}\tLoss: {:.6f}\tAccuracy: {:.6f}".format(
                    self.id, results["loss"], results["dice_coef"])
                )
        
        return parameters_prime, results