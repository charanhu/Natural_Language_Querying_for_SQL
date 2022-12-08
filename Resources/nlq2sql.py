# NLQ to SQL

# Import necessary modules
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from nlq_to_sql import NLQToSQL

# Connect to the database
db = sqlite3.connect("my_database.db")

# Define the database schema
schema = {
    "employees": ["id", "name", "department", "salary"],
    "departments": ["id", "name"]
}

# Initialize the NLQ to SQL system
nlq_to_sql = NLQToSQL()

# Train the NLQ to SQL system on the given database schema
nlq_to_sql.train(schema)

# Create the seq2seq model
model = Sequential()
model.add(Embedding(nlq_to_sql.vocab_size, 128, input_length=nlq_to_sql.max_input_length))
model.add(LSTM(128))
model.add(Dense(nlq_to_sql.vocab_size, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Ask a natural language question
question = "What are the names of the departments and the average salary of their employees?"

# Generate the corresponding SQL query
query = nlq_to_sql.predict(question)

# Execute the SQL query on the database
cursor = db.execute(query)

# Print the results
print("Query results:")
for row in cursor:
    print(row)

db.close()