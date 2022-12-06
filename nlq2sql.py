# wget http://nlp.stanford.edu/data/glove.6B.zip
# unzip glove.6B.zip

import names
import numpy as np
import tensorflow as tf

embeddings_index = {}
with open('glove.6B.100d.txt') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

fixed_embedding_matrix = np.zeros((vocab_size, 100))
for i, word in enumerate(vocabulary):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        fixed_embedding_matrix[i] = embedding_vector

fixed_embedding = tf.keras.layers.Embedding(
    self.nli_voba_size,
    100,
    embeddings_initializer=tf.keras.initializers.Constant(
        fixed_embedding_matrix),
    trainable=False,
    mask_zero=True)


class Nl2SqlTranslator(tf.keras.Model):
    def __init__(self, nl_text_processor, sql_text_processor, fixed_embedding, unit=128):
        super().__init__()
        # Natural language
        self.nl_text_processor = nl_text_processor
        self.nl_voba_size = len(nl_text_processor.get_vocabulary())
        self.nl_embedding = tf.keras.layers.Embedding(
            self.nl_voba_size,
            output_dim=unit,
            mask_zero=True)
        self.fixed_embedding = fixed_embedding
        self.nl_rnn = tf.keras.layers.Bidirectional(layer=tf.keras.layers.LSTM(
            int(unit/2), return_sequences=True, return_state=True))
        # Attention
        self.attention = tf.keras.layers.Attention()
        # SQL
        self.sql_text_processor = sql_text_processor
        self.sql_voba_size = len(sql_text_processor.get_vocabulary())
        self.sql_embedding = tf.keras.layers.Embedding(
            self.sql_voba_size,
            output_dim=unit,
            mask_zero=True)
        self.sql_rnn = tf.keras.layers.LSTM(
            unit, return_sequences=True, return_state=True)
        # Output
        self.out = tf.keras.layers.Dense(self.sql_voba_size)

    def call(self, nl_text, sql_text, training=True):
        nl_tokens = self.nl_text_processor(nl_text)  # Shape: (batch, Ts)
        # Shape: (batch, Ts, embedding_dim)
        nl_vectors = self.nl_embedding(nl_tokens, training=training)
        nl_fixed_vectors = self.fixed_embedding(
            nl_tokens)  # Shape: (batch, Ts, 100)
        # Shape: (batch, Ts, embedding_dim+100)
        nl_combined_vectors = tf.concat([nl_vectors, nl_fixed_vectors], -1)
        # Shape: (batch, Ts, bi_rnn_output_dim), (batch, rnn_output_dim) ...
        nl_rnn_out, fhstate, fcstate, bhstate, bcstate = self.nl_rnn(
            nl_vectors, training=training)
        nl_hstate = tf.concat([fhstate, bhstate], -1)
        nl_cstate = tf.concat([fcstate, bcstate], -1)

        sql_tokens = self.sql_text_processor(sql_text)  # Shape: (batch, Te)
        expected = sql_tokens[:, 1:]  # Shape: (batch, Te-1)

        teacher_forcing = sql_tokens[:, :-1]  # Shape: (batch, Te-1)
        # Shape: (batch, Te-1, embedding_dim)
        sql_vectors = self.sql_embedding(teacher_forcing, training=training)
        sql_in = self.attention(inputs=[sql_vectors, nl_rnn_out], mask=[
                                sql_vectors._keras_mask, nl_rnn_out._keras_mask], training=training)

        trans_vectors, _, _ = self.sql_rnn(sql_in, initial_state=[
                                           nl_hstate, nl_cstate], training=training)  # Shape: (batch, Te-1, rnn_output_dim)
        # Shape: (batch, Te-1, sql_vocab_size)
        out = self.out(trans_vectors, training=training)
        return out, expected, out._keras_mask


def populate(in_filename, out_filename):
    out = []
    with open(in_filename) as f:
        for l in f.readlines():
            parts = l.split('|||')
            nli_template = parts[0].strip().split()
            nli = []
            target = []
            for w in nli_template:
                if w == '[e_name_1]':
                    name = names.get_full_name().split()
                    nli.extend(name)
                    target.extend(['1' for n in name])
                elif w == '[e_salary_1]':
                    nli.append(str(random.randint(50000, 500000)))
                    target.append('2')
                elif w == '[e_salary_2]':
                    nli.append(str(random.randint(50000, 500000)))
                    target.append('3')
                elif w == '[e_gender_1]':
                    nli.append(random.choice(['male', 'female']))
                    target.append('4')
                else:
                    nli.append(w)
                    target.append('0')
            out.append(' ||| '.join([' '.join(nli), ' '.join(target)]))
    with open(out_filename, 'w') as f:
        for o in out:
            f.write(o + '\n')
