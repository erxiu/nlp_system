import re
import time
import tensorflow as tf


def preprocess_sentence(w):
    pass


def creat_dataset(file, num_examples):
    pass


if __name__ == '__main__':
    path_to_train_article = './data/train.article.txt'
    path_to_train_title = './data/train.title.txt'
    num_examples = 6400
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

    with open(path_to_train_article) as f:
        article_data = ['<start> ' + re.sub(r'[" "]+', " ", line.strip()) + ' <end>' for i, line in zip(range(num_examples), f)]
        lang_tokenizer.fit_on_texts(article_data)

    with open(path_to_train_title) as f:
        title_data = ['<start> ' + re.sub(r'[" "]+', " ", line.strip()) + ' <end>' for i, line in zip(range(num_examples), f)]
        lang_tokenizer.fit_on_texts(title_data)

    article_tensor = lang_tokenizer.texts_to_sequences(article_data)
    article_tensor = tf.keras.preprocessing.sequence.pad_sequences(article_tensor, padding='post')
    max_lenth_article = article_tensor.shape[1]

    title_tensor = lang_tokenizer.texts_to_sequences(title_data)
    title_tensor = tf.keras.preprocessing.sequence.pad_sequences(title_tensor, padding='post')
    max_lenth_title = title_tensor.shape[1]

    # Create tf.dataset
    BUFFER_SIZE = num_examples
    BATCH_SIZE = 64
    VOCAB_SIZE = len(lang_tokenizer.word_index) + 1
    steps_per_epoch = num_examples
    embedding_dim = 256
    units = 1024
    dataset = tf.data.Dataset.from_tensor_slices((article_tensor, title_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    example_input_batch, example_target_batch = next(iter(dataset))
    print('example_input_batch:', example_input_batch)
    print('example_target_batch:', example_target_batch)

    def convert(lang, tensor):
        for t in tensor:
            if t != 0:
                print("%d ----> %s" % (t, lang.index_word[t]))

    print("Input Language: index to word mapping")
    convert(lang_tokenizer, example_input_batch[0].numpy())
    print()
    print("Target Language: index to word mapping")
    convert(lang_tokenizer, example_target_batch[0].numpy())

    # Encoder
    class Encoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
            super(Encoder, self).__init__()
            self.batch_sz = batch_sz
            self.enc_units = enc_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

        def call(self, x, hidden):
            x = self.embedding(x)
            output, state = self.gru(x, initial_state=hidden)
            return output, state

        def initialize_hidden_state(self):
            return tf.zeros((self.batch_sz, self.enc_units))

    # Attention
    class BahdanauAttention(tf.keras.layers.Layer):
        def __init__(self, units):
            super(BahdanauAttention, self).__init__()
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)

        def call(self, query, values):
            query_with_time_axis = tf.expand_dims(query, 1)
            score = self.V((tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values))))
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = attention_weights * values
            context_vector = tf.reduce_sum(context_vector, axis=1)
            return context_vector, attention_weights

    # Decoder
    class Decoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
            super(Decoder, self).__init__()
            self.batch_sz = batch_sz
            self.dec_units = dec_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
            self.fc = tf.keras.layers.Dense(vocab_size)
            self.attention = BahdanauAttention(self.dec_units)

        def call(self, x, hidden, enc_output):
            context_vector, attention_weights = self.attention(hidden, enc_output)
            x = self.embedding(x)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
            output, state = self.gru(x)
            output = tf.reshape(output, (-1, output.shape[2]))
            x = self.fc(output)
            return x, state, attention_weights

    encoder = Encoder(VOCAB_SIZE, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(VOCAB_SIZE, embedding_dim, units, BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(inp, targ, enc_hidden):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    EPOCHS = 10
    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        # if (epoch + 1) % 2 == 0:
        #     checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch)*64)
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
