def generatorModelTrain(generator, model):
    results = model.fit(generator, steps_per_epoch=len(generator), epochs=200)
    prediction = model.predict(generator)
    return prediction
