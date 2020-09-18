import re
import tensorflow as tf


def preprocess_sentence(w):
    pass


def creat_dataset(file, num_examples):
    pass


if __name__ == '__main__':
    path_to_train_article = './data/train.article.txt'
    path_to_train_title = './data/train.title.txt'
    num_examples = 300
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

    with open(path_to_train_article) as f:
        article_data = ['<begin> ' + line.strip() + ' <end>' for i, line in zip(range(num_examples), f)]
        lang_tokenizer.fit_on_texts(article_data)
        article_tensor = lang_tokenizer.texts_to_sequences(article_data)
        article_tensor = tf.keras.preprocessing.sequence.pad_sequences(article_tensor, padding='post')
        max_lenth_article = article_tensor.shape[1]

    with open(path_to_train_title) as f:
        title_data = ['<begin> ' + line.strip() + ' <end>' for i, line in zip(range(num_examples), f)]
        lang_tokenizer.fit_on_texts(title_data)
        title_tensor = lang_tokenizer.texts_to_sequences(title_data)
        title_tensor = tf.keras.preprocessing.sequence.pad_sequences(title_tensor, padding='post')
        max_lenth_title = title_tensor.shape[1]

    # Create tf.dataset
    BUFFER_SIZE = num_examples
    BATCH_SIZE = 64
    steps_per_epoch = num_examples
    embedding_dim = 256
    units = 1024
    dataset = tf.data.Dataset.from_tensor_slices((article_tensor, title_tensor)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
