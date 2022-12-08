import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense


class NLQToSQL:
    def __init__(self):
        # Initialize the vocab size and max input length
        self.vocab_size = 100
        self.max_input_length = 20

        # Define the seq2seq model
        input_seq = Input(shape=(self.max_input_length,))
        lstm1 = LSTM(128, return_sequences=True)(input_seq)
        lstm2 = LSTM(128)(lstm1)
        dense = Dense(self.vocab_size, activation="softmax")(lstm2)
        self.model = Model(input_seq, dense)
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")

    def train(self, schema):
        # Initialize the input and output sequences
        input_seqs = []
        output_seqs = []

        # Create the input and output sequences for each question-query pair in the training data
        for question, query in training_data:
            # Encode the natural language question and SQL query
            input_seq = self.encode_input_seq(question)
            output_seq = self.encode_output_seq(query)

            # Add the input and output sequences to the list
            input_seqs.append(input_seq)
            output_seqs.append(output_seq)

        # Fit the seq2seq model on the input and output sequences
        self.model.fit(input_seqs, output_seqs, epochs=10, batch_size=32)

    def predict(self, question):
        # Encode the natural language question
        input_seq = self.encode_input_seq(question)

        # Use the seq2seq model to generate the SQL query
        query = self.model.predict(input_seq)

        # Decode the SQL query and return it
        return self.decode_output_seq(query)




########################################################
# same model as pickle file
import pickle

