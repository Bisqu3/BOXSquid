import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from keras.callbacks import TensorBoard

def generatorModelTrain(train_generator, test_generator, model, epoch_):
    #tensorboard setup
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

    #training the model
    results = model.fit(train_generator, steps_per_epoch=len(train_generator), 
                        epochs=epoch_, callbacks=[tensorboard_callback])
    
    #evaluating the model on test data
    evaluation_results = model.evaluate(test_generator)
    print("Evaluation Results:")
    print(f"Loss: {evaluation_results}")
    return results

