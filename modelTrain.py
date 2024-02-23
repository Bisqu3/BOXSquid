def generatorModelTrain(train_generator, test_generator, model,epoch_):
    #training the model
    results = model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=epoch_)
    
    #evaluating the model on test data
    evaluation_results = model.evaluate(test_generator)
    print("Evaluation Results:")
    print(f"Loss: {evaluation_results}")
    #Additional analysis below
    return results
