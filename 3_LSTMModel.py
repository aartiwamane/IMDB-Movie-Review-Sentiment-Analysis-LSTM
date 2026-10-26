#Step 1 : Import Required Libraries

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

#----------------------------------------------------------
#Step 2 : Configuration of values
#----------------------------------------------------------
VOCAB_SIZE = 10000      #Consider most frequent 10000 unique words
MAX_LENTH = 200         #Consider maximum 200 words in reviews

#----------------------------------------------------------
#Step 3 : Load the IMDB dataset
#----------------------------------------------------------
print("-"*40)
print("Movie Review Sentiment Analysis Using LSTM")
print("-"*40)

print("Loading the Dataset")
(X_train,Y_train),(X_test,Y_test) = imdb.load_data(num_words=VOCAB_SIZE)

print("IMDB Dataset Loaded successfully")

print("Number of training reviews : ",len(X_train))
print("Number of testing reviews : ",len(X_test))

#----------------------------------------------------------
#X_train        Reviews used for training
#Y_train        actual sentiments of training
#X_test         reviews used for training 
#Y_test         Actual sentiments of testing

#Sentiments :
#0 -> Negative sentiments
#1 -> Positive sentiments
#----------------------------------------------------------

#----------------------------------------------------------
#Step 4 : Load the word dictonary
#----------------------------------------------------------
word_index = imdb.get_word_index()

#Disctonary contains mapping of word and its corresponding number
#Drishyam is good movie     ->(20 56 78 43)
#20  -> Drishyam
#56  -> is
#78  -> good
#43   -> movie 

#----------------------------------------------------------
#Step 5 : Create Reverse dictonary
#----------------------------------------------------------

reverse_word_index = {}

for word,index in word_index.items():
    reverse_word_index[index+3] = word

#----------------------------------------------------------
#Step 6 : Function to Decode the review(number to word)
#----------------------------------------------------------
def DecodeReview(encoded_review):
    words = []

    for number in encoded_review:
        if number >= 3:  #ignore first three
            word = reverse_word_index.get(number,"?")
            words.append(word)

    return " ".join(words)      #join the list of words

#----------------------------------------------------------
#Step 7 : Display Sample Reviews 
#----------------------------------------------------------
print("-"*40)
print("----------------Sample Reviews----------------")
print("-"*40)

for i in range(3,7):
    review = DecodeReview(X_train[i])

    print("-"*40)

    print("Review number : ",i+1)
    print("Review : ")
    print(review)
    
    print("-"*40)


    if Y_train[i] == 1:
        print("Sentiment : Positive ")

    else:
        print("Sentiment : Negative ")

#----------------------------------------------------------
#Step 8 :Padding
#----------------------------------------------------------
X_train_Padded = pad_sequences(
    X_train,
    maxlen = MAX_LENTH
)

X_test_padded = pad_sequences(
    X_test,
    maxlen = MAX_LENTH
)

print("Training data shape : ",X_train_Padded.shape)
print("Testing data shape : ",X_test_padded.shape)

#----------------------------------------------------------
#Step 9 : Create LSTM Model
#----------------------------------------------------------

model = Sequential()

model.add(
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=32           #Each word is represented in 32 values
    )
)

model.add(
    LSTM(
        units = 64      #size of LSTM hidden state
    )
)

model.add(
    Dense(
        units=1,        #one output
        activation='sigmoid'    #used to produce probabilty
    )
)

#Project architecture
#Review -> Embedding -> LSTM -> Dense -> Sigmoid ->Positive/Negative

#----------------------------------------------------------
#Step 10 : Compile the model
#----------------------------------------------------------
model.compile(
    optimizer = "adam",      #algorithm to update weights
    loss = "binary_crossentropy",   #Loass function
    metrics = ["accuracy"]  #measure classification accuracy
)

print("Model compiled successfully")

#----------------------------------------------------------
#Step 11 : Train the model
#----------------------------------------------------------
print("MOdel Training")

model.fit(
    X_train_Padded,     #Input training reviews
    Y_train,            #Actual sentiment labels
    epochs = 3,         #Complete dataset gets processes 3 times
    batch_size = 64,     #Process 64 reviews in one batch 
    validation_split = 0.2 # use 20% training for validation
)
print("Model training gets completed")
#----------------------------------------------------------
#Step 12 : Evaluate the model
#----------------------------------------------------------
accuracy = model.evaluate(
    X_test_padded,      #Testing Reviews
    Y_test,             #Actual Testing Labels
    verbose = 0         #Don't Display progess bar
    )

print("Testing Accuracy : ",accuracy)

#----------------------------------------------------------
#Step 13 : Predict the review
#----------------------------------------------------------
TEST_REVIEW_NUMBER = 0

original_review = X_test[TEST_REVIEW_NUMBER]
decoded_review = DecodeReview(original_review)
print("Review given to the model : ")
print(decoded_review)

#----------------------------------------------------------
#Step 14 : Get the actual sentiment
#---------------------------------------------------------- 
actual_value = Y_test[TEST_REVIEW_NUMBER]

if actual_value == 1:
    actual_sentiment = "POSITIVE"
else:
    actual_sentiment = "NEGATIVE"

print("Actual Sentiment : ",actual_sentiment)

#----------------------------------------------------------
#Step 15 : Predict the sentiment
#----------------------------------------------------------
review_for_preiction = X_test_padded[TEST_REVIEW_NUMBER : TEST_REVIEW_NUMBER +1]

prediction = model.predict(
    review_for_preiction,
    verbose = 0
)

probability = prediction[0][0]

if probability >= 0.5:
    predicted_sentiment = "POSITIVE"
else :
    predicted_sentiment = "NEGATIVE"

print('-'*40)
print("FInal result")
print("-"*40)

print("Prediction probability : ",probability)
print("Predicted sentiment : ",predicted_sentiment)

print("-"*40)
