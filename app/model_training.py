
from ASL import *
from keras._tf_keras.keras.saving import save_model

# create_bins()
# data_collection2()
sequencing()
X=np.array(sequences)
y=to_categorical(labels).astype(int)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
log_dir=os.path.join("Logs")
tb_callback=TensorBoard(log_dir=log_dir)

try:
    model.fit(x_train, y_train, epochs=6000, callbacks=tb_callback)
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving model...")


save_model(model, "model2.keras")
print("Model saved to model2.keras")

