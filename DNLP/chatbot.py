# Chatbot using Deep NLP on tensorflow 1.0.0

# Building the chatbot

import numpy as np
import tensorflow as tf
import time 
import re # For cleaning the text/conversations

## Preprocessing
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n') # The split is to split the observations by lines
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
# Now we need to map each line of movie with ID
id2line= {}
for line in lines:
    _line =  line.split(' +++$+++ ') # Local variable only used in loop
    if (len(_line)== 5):
        id2line[_line[0]] = _line[4]
# We need to now make a list to keep track of the conversations 
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation =  conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","") # Local variable only used in loop
    conversations_ids.append(_conversation.split(","))

# Now we need to create the question and answer lists 
questions = []
answers = []
# We know that the answer to one question is the sentence with the next id
# Hence, we have that the answer to one id is the next id in the conversation_id list  
for conversation in conversations_ids:
    for i in range(len(conversation)- 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i + 1]])

# Now we need to clean the data, by defining all the sentences

def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text

# Cleaning the questions and answers
clean_questions = []
for question in questions:
    clean_questions.append(clean_txt(question))
    
clean_answers = []
for answer in answers:
    clean_answers.append(clean_txt(answer))
# Filtering out the questions and answers that are too short or too long
short_questions = []
short_answers = []
i = 0
for question in clean_questions:
    if 2 <= len(question.split()) <= 25:
        short_questions.append(question)
        short_answers.append(clean_answers[i])
    i += 1
clean_questions = []
clean_answers = []
i = 0
for answer in short_answers:
    if 2 <= len(answer.split()) <= 25:
        clean_answers.append(answer)
        clean_questions.append(short_questions[i])
    i += 1
# We need to remove the recurrent words
word2count = {}
for question in clean_questions:
    for word in question.split():
        if(word not in word2count):
            word2count[word] = 1
        else:
            word2count[word]+=1
# Now for the answer, we have 
for answer in clean_answers:
    for word in answer.split():
        if(word not in word2count):
            word2count[word] = 1
        else:
            word2count[word]+=1

# Now to do the tokenization of words,by mapping the words of questions and answers to integers
# We also have to filter out the no.of occurances below a threshold
# We generally filter out 5% of the words that occur the least
threshold = 20 # This is a hyperparameter
questionswords2int = {}
answerswords2int = {}
word_num = 0
for word,count in word2count.items():
    if(count >=threshold):
        questionswords2int[word] =  word_num # Unique integer 
        word_num +=1
# Now for the answers
word_num = 0
for word,count in word2count.items():
    if(count >=threshold):
        answerswords2int[word] =  word_num # Unique integer 
        word_num +=1

# Now to add the unique tokens
# We need to specify the out, EOS,SOS tokens
tokens = ['<PAD>','<EOS>', '<OUT>','<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1
# Now we need to create the inverse mapping of the integers to words(inverse the dictionary)
answersint2word =  {w_i:w for w,w_i in answerswords2int.items()}

# We need to add the EOS to each clean answers 
for i in range(len(clean_answers)):
    clean_answers[i] = clean_answers[i] + ' <EOS>'
    
# Now to convert the cleaned questions and answers into integer form and arrange them in length wise
# and then replacing the filtered words with the out tag.
questions_to_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if(word not in questionswords2int):
            ints.append(questionswords2int['<OUT>'])
        else :
            ints.append(questionswords2int[word])
    questions_to_int.append(ints)

answers_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if(word not in questionswords2int):
            ints.append(answerswords2int['<OUT>'])
        else :
            ints.append(answerswords2int[word])
    answers_to_int.append(ints)
# Now sorting the questions and answers by their length
# They are sorted based on the length of the questions because this helps in easy training 
sorted_clean_questions = []
sorted_clean_answers = []
# We don't need the sentences to be very big , hence we fix a specific length
# The length may be 25
for length in range(1,25 + 1):
    for i in enumerate(questions_to_int): # We will get the index as well as the length of the list by enumerate
        # The tuple i has the index as well as the entire row corresponding 
        if(length == len(i[1])):
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])
# Hence, we have the questions sorted and the answers corresponding to the questions too

## Building the RNN

# Creating the placeholders for the inputs
def model_input():
    inputs = tf.placeholder(tf.int32, [None,None], name = 'input') # The None,None is for specifying that our input data, the sorted_clean questions
    # is 2D
    targets = tf.placeholder(tf.int32, [None,None], name = 'target')
    # Learning rate and keep(drop out) paramters
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_drop')  # For the dropout control 
    return inputs, targets, lr,keep_prob

# Preprocessing the targets
# The targets must have special structure for them to be fed into the decoder and they should be in batches
# First the targets must be in batches(more than 1 answers), second  we need to attach the SOS token infront of all the tokens
# We remove the last column of the answers(EOS) and fix the SOS token in the front to preserve length
def preprocess_targets(target, word2int, batch_size):
    left_side = tf.fill([batch_size,1], word2int['<SOS>'])
    right_side = tf.strided_slice(target, [0,0], [batch_size,-1], [1,1])
    preprocessed_targets = tf.concat([left_side,right_side], axis = 1)
    return preprocessed_targets

# Creating the Stacke LSTM RNN for Encoder
def encoder_rnn(rnn_inputs, rnn_size, keep_prob, sequence_length, num_layers):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell( [lstm_dropout]*num_layers)
    _,encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell, cell_bw=encoder_cell, inputs = rnn_inputs, sequence_length = sequence_length, dtype = tf.float32)
    # The above encoder state gives the memory state of each rnn layer, and the bidirectional dynamic rnn gives the rnn layer with as many inputs as we want 
    # For stacking teh rnn, we have the numtirnncell fn from tf.contrib.rnn
    return encoder_state

# Decoding the training set

# Embedding is the convertion of the words to vectors of real numbers, that the decoder takes in as the input
# Now the variable scope is for wrapping the tensor variables 
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, decoding_scope, output_function, keep_prob, batch_size):
    # The output_fucntion is fo rreturning the decoder output 
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size]) # This gives a three dimentional tensor for the decoder cell to serve as input
    # Now we need to make the attention key, values, score function and contruct function 
    # This is to prepare the input for the attention process
    attention_keys, attention_values, attention_score_function, attention_contruct_function = tf.contrib.seq2seq.prepare_attention(attention_states,attention_option='bahdanau', num_units=decoder_cell.output_size )
    # attention_keys are to be compared with the target states, the context is returned by the encoder and they are the attention_values,
    # the attention_score is for comparning the similarity of keys and target states
    # contruct for building the attention state
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], attention_keys, attention_values,attention_score_function,attention_contruct_function, name='attn_dec_train')
    #  Nmae scope is the tensor name for the decoder function
    # Now we use the dynamic_rnn_decoder to contruct the decoder
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(cell=decoder_cell, decoder_fn=training_decoder_function,
                                                                                                              inputs=decoder_embedded_input,sequence_length=sequence_length,
                                                                                                              scope= decoding_scope)
    # We only need the decoder_output
    decoder_output_dropout = tf.nn.dropout(decoder_output,keep_prob)
    # The decoder output is added with some dropout using the ordinary method of dropout addition
    return output_function(decoder_output_dropout)

# Decoding the test/validation set
# The validation set is the set that we make during the training, for crosss validation of the results after the training 

def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    # The output_fucntion is fo rreturning the decoder output 
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size]) # This gives a three dimentional tensor for the decoder cell to serve as input
    # Now we need to make the attention key, values, score function and contruct function 
    # This is to prepare the input for the attention process
    # The decoder_embeddings_matrix is not same as that of a decoder_embedded_inputs
    attention_keys, attention_values, attention_score_function, attention_contruct_function = tf.contrib.seq2seq.prepare_attention(attention_states,attention_option='bahdanau', num_units=decoder_cell.output_size )
    # attention_keys are to be compared with the target states, the context is returned by the encoder and they are the attention_values,
    # the attention_score is for comparning the similarity of keys and target states
    # contruct for building the attention state
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                                                              encoder_state[0],
                                                                                                              attention_keys, attention_values,
                                                                                                              attention_score_function,
                                                                                                              attention_contruct_function,
                                                                                                              decoder_embeddings_matrix,
                                                                                                              sos_id,
                                                                                                              eos_id,
                                                                                                              maximum_length,
                                                                                                              num_words,
                                                                                                              name='attn_dec_inf')
                                                                                                              
    #  Nmae scope is the tensor name for the decoder function
    # Now we use the dynamic_rnn_decoder to contruct the decoder
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(cell=decoder_cell, decoder_fn=test_decoder_function,
                                                                                                              scope= decoding_scope)
    # We only need the decoder_output

    return test_predictions

# Now, to contruct the Decoder RNN, like the Encoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    # decoding scope should be introduced, which is actually the tensor variable which contains lots of data
    with tf.variable_scope('decoding') as decoding_scope:
        # This is a scope
        # Declare LSTM layer
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell( [lstm_dropout]*num_layers)
        # We need to initialize the weights of the fully connected layer of the rnn
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        # We need the fully connected layer at the end of the stacked lstm layers
        output_function = lambda x: tf.contrib.layers.fully_connected(x,num_words, normalizer_fn = None, scope= decoding_scope,
                                                                      weights_initializer=weights,
                                                                      biases_initializer=biases)
        # This output function is the fully connected layer, which takes the input from the previous stacked lstm
        
        # Now we need to get teh training set preictions
        training_predictions = decode_training_set(encoder_state, decoder_cell,
                                                   decoder_embedded_input,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        # Now for the test predictions
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length-1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions,test_predictions

# Building the Seq2Seq model

def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    # decoder_embedding_size and encoder is the dimentions of the embedded matrix for the encoder and the decoder
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              tf.random_normal_initializer(0,1))
    encoder_state = encoder_rnn(encoder_embedded_input,rnn_size,keep_prob,sequence_length,num_layers)
    # Now we need the preprocessed targets and the embedded inputs for the decoder from the embedding matrix
    preprocessed_targets = preprocess_targets(targets,questionswords2int,batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1,encoder_embedding_size],0,1)) # Initialize the matrix with random normalized numbers
    # We have the decoder_embeddings_matrix having the words in line of rows and the embeddings of their corresponding words in the column
    decoder_embedded_inputs = tf.nn.embedding_lookup(decoder_embeddings_matrix,preprocessed_targets,)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_inputs,decoder_embeddings_matrix,encoder_state,
                                                         questions_num_words,sequence_length,
                                                         rnn_size,num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions

## Training the Seq2Seq Model

# Selecting the hyperparameter
epochs = 100
# What happens is that we input the encoder embedded inputs and then we take the encoder state and the nwe input these into the decoder and then we obtain the output predictions 
batch_size = 64
rnn_size= 512
num_layers= 3
encoder_embedding_size = 512 # The number of columns in the embedding matrix , and the column giving the number of column in the embeddings matrix
decoder_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9 # For reducing the learning rate as we train 
min_learning_rate = 0.0001 # For having the minium learning for the model if it gets very low
keep_probability = 0.5 # this is 1-dropout rate

# we define a tensorflow session
# we need to reset the graphs before training in the session
tf.reset_default_graph()
session = tf.InteractiveSession() # Defined a session

# Load the model inputs
inputs, targets, lr, keep_prob =model_input()
# Setting the sequence length 
sequence_length = tf.placeholder_with_default(25,None, name= 'sequence_length')

# Setting the shape of the input tensor
input_shape = tf.shape(inputs)
# Getting the training predictions and the test predictions
# we need to get the predictions of the model when we are inputing the above inputs,targets and lr and keep_prob
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs,[-1]),targets,keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoder_embedding_size,
                                                       decoder_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)
# We are setting the loss error , optimizer and gradient clipping

with tf.name_scope('optimization'):
    loss_error= tf.contrib.seq2seq.sequence_loss(training_predictions, targets, 
                                                 tf.ones([input_shape[0], sequence_length])) # between the training predictions and the targets
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradient = optimizer.compute_gradients(loss_error)
    clipper_gradients = [ (tf.clip_by_value(grad_tensor, -5., 5.),grad_variable) for grad_tensor,grad_variable in gradient if grad_tensor is not None] # we have to loop through the gradients for ensuring that the gradient is bound
    optimizer_gradient_clipping = optimizer.apply_gradients(clipper_gradients)

# Padding the sequence with the pad token <PAD>
# The pad tokens are added in both questions and answers so that the size of both questions and answers are of same length
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [ sequence + [word2int['<PAD>']]*(max_sequence_length-len(sequence)) for sequence in batch_of_sequences]

# Splitting the data into batches of questions and answers 
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0,len(questions)//batch_size):
        start_index = batch_index*batch_size # This is the index of the first question/answer in the batch
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch,questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch,answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch
        
# Splitting the answers and questions into training and validation set
training_validation_split = int(len(sorted_clean_questions)*0.15)
training_questions = sorted_clean_questions[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
training_answers = sorted_clean_answers[training_validation_split:]
validation_answers = sorted_clean_answers[:training_validation_split]

## Now we start the training of the seq2seq model
batch_index_check_training_loss =100 
batch_index_check_validation_loss = (len(training_questions)//batch_size//2 ) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index,(padded_questions_in_batch,padded_answers_in_batch) in enumerate(split_into_batches(training_questions,training_answers,batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping,loss_error], {inputs: padded_questions_in_batch,
                                                                                              targets : padded_answers_in_batch, lr : learning_rate,
                                                                                              sequence_length: padded_answers_in_batch.shape[1],
                                                                                              keep_prob:keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        # Now we need to get the training loss error at the end of every batch_index_check_training_loss
        if (batch_index % batch_index_check_training_loss == 0):
            print("Epoch: {:>3}/{}, Batch: {:>4}/{}, Training loss error: {:>6.3f}, Training time for 100 batches: {:d} seconds".format(epoch, epochs,batch_index,
                                                                                                                                           len(training_questions)//batch_size,
                                                                                                                                           total_training_loss_error/batch_index_check_training_loss ,
                                                                                                                                           int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if (batch_index % batch_index_check_validation_loss == 0 and batch_index >0):
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation,(padded_questions_in_batch,padded_answers_in_batch) in enumerate(split_into_batches(validation_questions,validation_answers,batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets : padded_answers_in_batch, lr : learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob:1})
                                                                                              
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error/ (len(validation_questions)/batch_size)
            print("Validation loss error: {:>6.3f}, Batch Validation time : {:d} seconds".format(average_validation_loss_error,int(batch_time)))
            learning_rate *= learning_rate_decay
            if(learning_rate < min_learning_rate):
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if(average_validation_loss_error <=list_validation_loss_error):
                print("I speack better now!!")
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session,checkpoint)
            else:
                print("Sorry, I do not speak better, I need to practice more")
                early_stopping_check +=1
                if(early_stopping_check == early_stopping_stop):
                    break
    if(early_stopping_check == early_stopping_stop):
        print("My apologies, I cannot speak better anymore. This is the best i can do")
        break
print("Game Over")

# Now testing the bot 

# Now, after the training, we need to initialize the weights and load the model
# loading the weights from the checkpoint
checkpoint = "./ <file_name>"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
# we connect the weights to the session
saver = tf.train.Saver()
saver.restore(session,checkpoint)

# Now to make a function to convert the words into encoded integers
def convert_string2int(question,word2int):
    question = clean_txt(question)
    return [word2int.get(word,word2int['<OUT>']) for word in question.split()] # this is to make sure that the frequesntly used words are only included

# Setting the chat 
while(True):
    question = input("You: ")
    if (question == "goodbye"):
        break
    question = convert_string2int(question,questionswords2int)
    question = question + [questionswords2int['<PAD>']]* (20 - len(question))
    fake_batch = np.zeros((batch_size,20))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer,1):
        if answersint2word[i] == 'i':
            token = 'I'
        elif answersint2word[i] == '<EOS>':
            token = "."
        elif answersint2word[i] == '<OUT>':
            token = "out"
        else:
            token = ' ' + answersint2word[i]
        answer += token
        
        if token== '.':
            break
    print("Chatbot: " + answer)
















    
    
    