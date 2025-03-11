import tensorflow as tf
from matplotlib import pyplot as plt

import layer_encoding
import transformer_layer
import background as bd
import callBack
import data_cleaning

# Define loss function based on relative residual
def loss_fn(labels, targets):
    labels_mask = tf.math.is_nan(labels)
    labels = tf.where(labels_mask, tf.zeros_like(labels), labels)
    mask_count = tf.math.reduce_sum(tf.where(labels_mask, tf.zeros_like(labels), tf.ones_like(labels)))
    loss = tf.math.abs(labels - targets)
    loss = tf.where(labels_mask, tf.zeros_like(loss), loss)
    loss = tf.math.reduce_sum(loss)/mask_count
    return loss


# tried hidden_dim = 192
def get_model(hidden_dim = 384, max_len = 206):
    with bd.strategy.scope():
        inp = tf.keras.Input([max_len])
        x = inp

        x = tf.keras.layers.Embedding(bd.num_vocab, hidden_dim, mask_zero=True)(x)
        x = layer_encoding.positional_encoding_layer(num_vocab=bd.num_vocab,
                                                     maxlen=500,
                                                     hidden_dim=hidden_dim)(x)

        # define depth of model, depth = 6, 8, 10, 15, 20
        for _ in range(10):
            x = transformer_layer.transformer_block(hidden_dim, 6, hidden_dim * 4)(x)

        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(2)(x)

        model = tf.keras.Model(inp, x)
        loss = loss_fn
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005)
        model.compile(loss=loss, optimizer=optimizer, steps_per_execution=100)
        return model


tf.keras.backend.clear_session()

model = get_model(hidden_dim=192, max_len=bd.X_max_len)
model(data_cleaning.batch[0])

model.summary() # with total param 5,339,714, divided into Embedding (None, 206, 192), each layer deals 444864 param

# Training the model with a callback to output a few transcriptions
steps_per_epoch = data_cleaning.num_train//bd.batch_size
val_steps_per_epoch = data_cleaning.num_val//bd.val_batch_size
print(steps_per_epoch)
print(val_steps_per_epoch)

history = model.fit(
    data_cleaning.train_dataset,
    validation_data=data_cleaning.val_dataset,
    epochs=callBack.N_EPOCHS,
    steps_per_epoch = steps_per_epoch,
    validation_steps=val_steps_per_epoch,
    verbose = 2,
    callbacks=[
        callBack.save_model_callback(),
        callBack.lr_callback,
    ]
)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

