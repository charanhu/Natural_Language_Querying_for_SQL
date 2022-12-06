# Import NLQ to SQL module
from nlq_to_sql import NLQToSQL

# Initialize NLQ to SQL system
nlq_to_sql = NLQToSQL()

# Define a sample database schema
schema = {
    "customers": ["id", "name", "email"],
    "orders": ["id", "customer_id", "price"]
}

# Train NLQ to SQL system on the given database schema
nlq_to_sql.train(schema)

# Ask a natural language question
question = "What are the names of the customers who made orders?"

# Generate the corresponding SQL query
query = nlq_to_sql.predict(question)

# Print the generated SQL query
print(query)  # SELECT name FROM customers WHERE id IN (SELECT customer_id FROM orders)



##################################################################################################################

# Import NLQ to SQL module
from nlq_to_sql import NLQToSQL

# Initialize NLQ to SQL system
nlq_to_sql = NLQToSQL()

# Define a sample database schema
schema = {
    "employees": ["id", "name", "department", "salary"],
    "departments": ["id", "name"]
}

# Train NLQ to SQL system on the given database schema
nlq_to_sql.train(schema)

# Ask a natural language question
question = "What are the names of the departments and the average salary of their employees?"

# Generate the corresponding SQL query
query = nlq_to_sql.predict(question)

# Print the generated SQL query
print(query)  # SELECT departments.name, AVG(employees.salary) FROM departments INNER JOIN employees ON departments.id = employees.department GROUP BY departments.name


#############################################################################################################################

# Import NLQ to SQL module
from nlq_to_sql import NLQToSQL

# Initialize NLQ to SQL system
nlq_to_sql = NLQToSQL()

# Define a sample database schema
schema = {
    "students": ["id", "name", "age", "grade"],
    "teachers": ["id", "name", "department"],
    "courses": ["id", "name", "teacher_id"],
    "enrollments": ["id", "student_id", "course_id", "grade"],
    "departments": ["id", "name"],
    "scores": ["id", "enrollment_id", "score"],
    "attendance": ["id", "enrollment_id", "attendance_date", "present"]
}

# Train NLQ to SQL system on the given database schema
nlq_to_sql.train(schema)

# Ask a natural language question
question = "What are the names of the teachers who teach courses that have at least one student who scored above 90 in all exams?"

# Generate the corresponding SQL query
query = nlq_to_sql.predict(question)

# Print the generated SQL query
print(query)  # SELECT DISTINCT teachers.name FROM teachers INNER JOIN courses ON teachers.id = courses.teacher_id INNER JOIN enrollments ON courses.id = enrollments.course_id INNER JOIN scores ON enrollments.id = scores.enrollment_id WHERE scores.score > 90 GROUP BY teachers.name HAVING COUNT(DISTINCT scores.score) = COUNT(DISTINCT scores.enrollment_id)


#############################################################################################################################

# Import necessary modules
import tensorflow as tf
from nlq_to_sql import NLQToSQL

# Define a sample database schema
schema = {
    "employees": ["id", "name", "department", "salary"],
    "departments": ["id", "name"]
}

# Initialize the NLQ to SQL system with TensorFlow
nlq_to_sql = NLQToSQL(tf)

# Train the NLQ to SQL system on the given database schema
nlq_to_sql.train(schema)

# Ask a natural language question
question = "What are the names of the departments and the average salary of their employees?"

# Generate the corresponding SQL query
query = nlq_to_sql.predict(question)

# Print the generated SQL query
print(query)  # SELECT departments.name, AVG(employees.salary) FROM departments INNER JOIN employees ON departments.id = employees.department GROUP BY departments.name


#############################################################################################################################

# Import necessary modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from nlq_to_sql import NLQToSQL

# Define a sample database schema
schema = {
    "students": ["id", "name", "age", "grade"],
    "teachers": ["id", "name", "department"],
    "courses": ["id", "name", "teacher_id"],
    "enrollments": ["id", "student_id", "course_id", "grade"],
    "departments": ["id", "name"],
    "scores": ["id", "enrollment_id", "score"],
    "attendance": ["id", "enrollment_id", "attendance_date", "present"]
}

# Initialize the NLQ to SQL system
nlq_to_sql = NLQToSQL()

# Train the NLQ to SQL system on the given database schema
nlq_to_sql.train(schema)

# Create the seq2seq model
model = Sequential()
model.add(Embedding(nlq_to_sql.vocab_size, 128, input_length=nlq_to_sql.max_input_length))
model.add(LSTM(128))
model.add(Dense(nlq_to_

#############################################################################################################################


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


#############################################################################################################################


# Import necessary modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from nlq_to_sql import NLQToSQL

# Define a sample database schema
schema = {
    "students": ["id", "name", "age", "grade"],
    "teachers": ["id", "name", "department"],
    "courses": ["id", "name", "teacher_id"],
    "enrollments": ["id", "student_id", "course_id", "grade"],
    "departments": ["id", "name"],
    "scores": ["id", "enrollment_id", "score"],
    "attendance": ["id", "enrollment_id", "attendance_date", "present"]
}

# Initialize the NLQ to SQL system with TensorFlow
nlq_to_sql = NLQToSQL(tf)

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

# Print the generated SQL query
print(query)  # SELECT departments.name, AVG(employees.salary) FROM departments INNER JOIN employees ON departments.id = employees.department GROUP BY departments.name


#############################################################################################################################

# Import necessary modules
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from nlq_to_sql import NLQToSQL

# Define a sample database schema
schema = {
    "students": ["id", "name", "age", "grade"],
    "teachers": ["id", "name", "department"],
    "courses": ["id", "name", "teacher_id"],
    "enrollments": ["id", "student_id", "course_id",


#############################################################################################################################

# Import necessary modules
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from nlq_to_sql import NLQToSQL

# Define a sample database schema
schema = {
    "students": ["id", "name", "age", "grade"],
    "teachers": ["id", "name", "department"],
    "courses": ["id", "name", "teacher_id"],
    "enrollments": ["id", "student_id", "course_id", "grade"],
    "departments": ["id", "name"],
    "scores": ["id", "enrollment_id", "score"],
    "attendance": ["id", "enrollment_id", "attendance_date", "present"]
}

# Initialize the NLQ to SQL system with TensorFlow
nlq_to_sql = NLQToSQL(tf)

# Train the NLQ to SQL system on the given database schema
nlq_to_sql.train(schema)

# Create the seq2seq model
model = Sequential()
model.add(Embedding(nlq_to_sql.vocab_size, 128, input_length=nlq_to_sql.max_input_length))
model.add(LSTM(128))
model.add(Dense(nlq_to_sql


#############################################################################################################################

# Import necessary modules
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from nlq_to_sql import NLQToSQL

# Define a sample database schema
schema = {
    "students": ["id", "name", "age", "grade"],
    "teachers": ["id", "name", "department"],
    "courses": ["id", "name", "teacher_id"],
    "enrollments": ["id", "student_id", "course_id", "grade"],
    "departments": ["id", "name"],
    "scores": ["id", "enrollment_id", "score"],
    "attendance": ["id", "enrollment_id", "attendance_date", "present"]
}

# Initialize the NLQ to SQL system with TensorFlow
nlq_to_sql = NLQToSQL(tf)

# Train the NLQ to SQL system on the given database schema
nlq_to_sql.train(schema)

# Create the seq2seq model
model = Sequential()
model.add(Embedding(nlq_to_sql.vocab_size, 128, input_length=nlq_to_sql.max_input_length))
model.add(LSTM(128))
model.add(Dense(nlq_to_sql.vocab_size, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Initialize the Flask app
app = Flask(__name__)

# Define the route for the NLQ to SQL web service
@app.route("/nlq-to-sql", methods=["POST"])
def nlq_to_sql_route():
    # Parse the natural language question from the request body
    question = request.json["question"]

    # Generate the corresponding SQL query
    query = nlq_to_sql.predict(question)

    # Return the generated SQL query in the response
    return jsonify({"query": query})

# Start the Flask app
if __name__ == "__main__":
    app.run()


#############################################################################################################################