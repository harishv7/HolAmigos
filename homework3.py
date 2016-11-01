import numpy as np
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import copy
import string

def convert_to_present_tense(word):
    return WordNetLemmatizer().lemmatize(word,'v')

def generate_permutations(subjects, verbs, nouns):
    permutations = []
    
    for subject in subjects:
        for verb in verbs:
            for noun in nouns:
                permutations.append((subject, verb, noun))
                
    return permutations

def split_index(data_point):
    tokens = data_point.split()
    index = tokens[0]
    content = " ".join(tokens[1:])
    return (index, content)

def parse_statement(statement):
    statement_tokens = word_tokenize(statement)
    # Do POS tagging on each word in the statement
    pos_tagged_statement = nltk.pos_tag(statement_tokens)
    
    # print(pos_tagged_statement)
    
    # Convert all verbs in the statement into simple present
    for i in range(len(pos_tagged_statement)):
        if ("VB" in pos_tagged_statement[i][1]):
            pos_tagged_statement[i] = (convert_to_present_tense(pos_tagged_statement[i][0]), "VB")
            
    # print(pos_tagged_statement)
    
    statement_subjects = []
    statement_verbs = []
    statement_nouns = []
    for i in range(len(pos_tagged_statement)):
        if ("NNP" in pos_tagged_statement[i][1]):
            statement_subjects.append(pos_tagged_statement[i][0])
        elif ("VB" in pos_tagged_statement[i][1]):
            statement_verbs.append(pos_tagged_statement[i][0])
        elif ("NN" in pos_tagged_statement[i][1]):
            statement_nouns.append(pos_tagged_statement[i][0])
            
    svn_permutations = generate_permutations(statement_subjects, statement_verbs, statement_nouns)
    
    return svn_permutations

def parse_qna(qna):
    qna_tokens = qna.split("?")
    question = qna_tokens[0].strip()
    answer = qna_tokens[1].strip()
    
    # Parse the question
    question_tokens = word_tokenize(question)
    pos_tagged_question = nltk.pos_tag(question_tokens)
    # print(pos_tagged_question)
    
    # Convert all verbs in the question into simple present
    for i in range(len(pos_tagged_question)):
        if ("VB" in pos_tagged_question[i][1]):
            pos_tagged_question[i] = (convert_to_present_tense(pos_tagged_question[i][0]), "VB")
            
    # print(pos_tagged_question)
    
    question_subjects = []
    question_verbs = []
    question_nouns = []
    for i in range(len(pos_tagged_question)):
        if ("NNP" in pos_tagged_question[i][1]):
            question_subjects.append(pos_tagged_question[i][0])
        elif ("VB" in pos_tagged_question[i][1]):
            question_verbs.append(pos_tagged_question[i][0])
        elif ("NN" in pos_tagged_question[i][1]):
            question_nouns.append(pos_tagged_question[i][0])
    
    # Parse the answer
    answer_tokens = word_tokenize(answer)
    pos_tagged_answer = nltk.pos_tag(answer_tokens)
    # print(pos_tagged_answer)
    
    # Convert all verbs in the answer into simple present
    for i in range(len(pos_tagged_answer)):
        if ("VB" in pos_tagged_answer[i][1]):
            pos_tagged_answer[i] = (convert_to_present_tense(pos_tagged_answer[i][0]), "VB")
            
    # print(pos_tagged_answer)
    
    answer_subjects = []
    answer_verbs = []
    answer_nouns = []
    for i in range(len(pos_tagged_answer)):
        if ("NNP" in pos_tagged_answer[i][1]):
            answer_subjects.append(pos_tagged_answer[i][0])
        elif ("VB" in pos_tagged_answer[i][1]):
            answer_verbs.append(pos_tagged_answer[i][0])
        elif ("NN" in pos_tagged_answer[i][1]):
            answer_nouns.append(pos_tagged_answer[i][0])
            
    return (question_subjects + question_verbs + question_nouns, answer_subjects + answer_verbs + answer_nouns)

def calculate_hash(string_to_hash):
    string_to_hash = string_to_hash.lower()
    
    hash_value = 0
    for ch in string_to_hash:
        hash_value *= 31
        hash_value += ord(ch) - ord('a') + 1
        hash_value %= 1000000007
        
    return hash_value

def main():
    NUM_TRAINING_SAMPLES = 100000
    
    training_dataset = pd.read_csv("train.txt", delimiter = "\n", header = None)
    
    print("Identifying all keywords and their permutations...")
    
    ctr = 0
    subjects = set()
    verbs = set()
    nouns = set()
    svn = set()
    for training_data_point in training_dataset[0]:
        index_content_pair = split_index(training_data_point)
        index = index_content_pair[0]
        content = index_content_pair[1]

        content_tokens = word_tokenize(content)
        pos_tagged_content = nltk.pos_tag(content_tokens)
        # print(pos_tagged_content)

        # Convert all verbs in the content into simple present
        for i in range(len(pos_tagged_content)):
            if ("VB" in pos_tagged_content[i][1]):
                pos_tagged_content[i] = (convert_to_present_tense(pos_tagged_content[i][0]), "VB")

        # print(pos_tagged_content)

        for i in range(len(pos_tagged_content)):
            if ("NNP" in pos_tagged_content[i][1]):
                subjects.add(pos_tagged_content[i][0])
            elif ("VB" in pos_tagged_content[i][1]):
                verbs.add(pos_tagged_content[i][0])
            elif ("NN" in pos_tagged_content[i][1]):
                nouns.add(pos_tagged_content[i][0])
        
        ctr += 1
        if (ctr % 1000 == 0):
            print(str(ctr) + " / " + str(len(training_dataset[0])))
        if (ctr == NUM_TRAINING_SAMPLES):
            break
            
    svn_permutations = generate_permutations(subjects, verbs, nouns)

    print("Number of subjects: " + str(len(subjects)))
    for subject in subjects:
        print(subject)
    print("Number of verbs: " + str(len(verbs)))
    for verb in verbs:
        print(verb)
    print("Number of nouns: " + str(len(nouns)))
    for noun in nouns:
        print(noun)
    print("Number of permutations: " + str(len(svn_permutations)))
    
    print("Identifying all the required hashes...")
    
    hash_to_input_neuron_id_map = {}
    current_input_neuron_id = 0
    hash_to_output_neuron_id_map = {}
    current_output_neuron_id = 0
    output_neuron_id_to_keyword_map = {}
    
    # Generate hash to input neuron id mapping and hash to output neuron id mapping
    for svn_permutation in svn_permutations:
        hash_value = 0
        for svn_permutation_element in svn_permutation:
            hash_value += calculate_hash(svn_permutation_element)
            hash_value %= 1000000007
        hash_to_input_neuron_id_map[hash_value] = current_input_neuron_id
        current_input_neuron_id += 1
    
    for subject in subjects:
        hash_value = calculate_hash(subject)
        hash_to_input_neuron_id_map[hash_value] = current_input_neuron_id
        hash_to_output_neuron_id_map[hash_value] = current_output_neuron_id
        output_neuron_id_to_keyword_map[current_output_neuron_id] = subject
        
        current_input_neuron_id += 1
        current_output_neuron_id += 1
        
    for verb in verbs:
        hash_value = calculate_hash(verb)
        hash_to_input_neuron_id_map[hash_value] = current_input_neuron_id
        hash_to_output_neuron_id_map[hash_value] = current_output_neuron_id
        output_neuron_id_to_keyword_map[current_output_neuron_id] = verb
        
        current_input_neuron_id += 1
        current_output_neuron_id += 1
        
    for noun in nouns:
        hash_value = calculate_hash(noun)
        hash_to_input_neuron_id_map[hash_value] = current_input_neuron_id
        hash_to_output_neuron_id_map[hash_value] = current_output_neuron_id
        output_neuron_id_to_keyword_map[current_output_neuron_id] = noun
        
        current_input_neuron_id += 1
        current_output_neuron_id += 1
        
    input_dimension = len(svn_permutations) + len(subjects) + len(verbs) + len(nouns)
    output_dimension = len(subjects) + len(verbs) + len(nouns)
    
    print("Number of input dimensions: " + str(input_dimension))
    print("Number of output dimensions: " + str(output_dimension))
    
    print("Extracting training inputs and expected outputs...")
    
    sample_inputs = []
    expected_outputs = []
    
    current_input = [0.0] * input_dimension
    current_output = [0.0] * output_dimension
    
    ctr = 0
    hashed_statements = []
    for training_data_point in training_dataset[0]:
        index_content_pair = split_index(training_data_point)
        index = index_content_pair[0]
        content = index_content_pair[1]

        # print(index)
        
        if (index == "1"):
            # Reset current input array
            current_input = [0.0] * input_dimension
            hashed_statements.clear()

        if ("?" in training_data_point):
            question_answer_pair = parse_qna(content)
            question_keywords = question_answer_pair[0]
            answer_keywords = question_answer_pair[1]
            
            s_id = 0.0
            for hashed_statement in hashed_statements:
                s_id += 1.0
                current_input[hash_to_input_neuron_id_map[hashed_statement]] = s_id / len(hashed_statements)
            
            for question_keyword in question_keywords:
                hash_value = calculate_hash(question_keyword)
                current_input[hash_to_input_neuron_id_map[hash_value]] = 1.0
                
            for answer_keyword in answer_keywords:
                hash_value = calculate_hash(answer_keyword)
                current_output[hash_to_output_neuron_id_map[hash_value]] = 1.0
            
            sample_inputs.append(copy.deepcopy(current_input))
            expected_outputs.append(copy.deepcopy(current_output))
            
            for question_keyword in question_keywords:
                hash_value = calculate_hash(question_keyword)
                current_input[hash_to_input_neuron_id_map[hash_value]] = 0.0
                
            current_output = [0.0] * output_dimension
        else:
            statement_permutations = parse_statement(content)
            
            for statement_permutation in statement_permutations:
                hash_value = (calculate_hash(statement_permutation[0]) + 
                              calculate_hash(statement_permutation[1]) + 
                              calculate_hash(statement_permutation[2])) % 1000000007
                hashed_statements.append(hash_value)
                
        # print()

        ctr += 1
        if (ctr % 1000 == 0):
            print(str(ctr) + " / " + str(len(training_dataset[0])))
        if (ctr == NUM_TRAINING_SAMPLES):
            break
    
    np.set_printoptions(threshold=np.nan)
    np.save("expected_output", np.array(expected_outputs))
    
    # Search for optimal hyperparameters by using grid search
    m_alpha = [1e-03]
    m_batch_size = [500]
    m_hidden_layer_sizes = [(400, 200,)]
    m_learning_rate_init = [0.003]
    
    for c_alpha in m_alpha:
        for c_batch_size in m_batch_size:
            for c_hidden_layer_sizes in m_hidden_layer_sizes:
                for c_learning_rate_init in m_learning_rate_init:
                    # Train the model
                    print("Training model with alpha = " + str(c_alpha) + 
                          ", batch size = " + str(c_batch_size) + 
                          ", hidden layer sizes = " + str(c_hidden_layer_sizes) + 
                          ", learning rate = " + str(c_learning_rate_init))
                    
                    average_accuracy = 0
                    for iteration in range(5):
                        # Initialize model
                        model = MLPClassifier(activation='logistic', alpha=c_alpha, batch_size=c_batch_size, 
                                              beta_1=0.9, beta_2=0.999, early_stopping=False,
                                              epsilon=1e-08, hidden_layer_sizes=c_hidden_layer_sizes, learning_rate='adaptive', 
                                              learning_rate_init=c_learning_rate_init, max_iter=1000, momentum=0.9,
                                              nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                                              solver='adam', tol=1e-04, validation_fraction=0.1, verbose=False,
                                              warm_start=False)
                        
                        # Shuffle training inputs and expected outputs in unison
                        sample_inputs, expected_outputs = shuffle(sample_inputs, expected_outputs, random_state = 0)
                        
                        model.fit(sample_inputs[:int(0.8 * len(sample_inputs))], expected_outputs[:int(0.8 * len(expected_outputs))])

                        # Cross-validate the model
                        np.set_printoptions(threshold=np.nan)
                        np.save("training_output", model.predict(sample_inputs[int(0.8 * len(sample_inputs)):]))

                        loaded_expected_outputs = expected_outputs[int(0.8 * len(expected_outputs)):]
                        loaded_training_outputs = np.load("training_output.npy")

                        hit = 0
                        miss = 0
                        for i in range(len(loaded_expected_outputs)):
                            if np.array_equal(loaded_expected_outputs[i], loaded_training_outputs[i]):
                                hit += 1
                            else:
                                miss += 1
                        accuracy = hit / (hit + miss)
                        print("Validation accuracy: " + str(accuracy))
                        average_accuracy += accuracy
                    
                    average_accuracy /= 5
                    print("Average validation accuracy: " + str(average_accuracy))
    
    print("Training the model...")
    model = MLPClassifier(activation='logistic', alpha=1e-03, batch_size=500, 
        beta_1=0.9, beta_2=0.999, early_stopping=False, 
        epsilon=1e-08, hidden_layer_sizes=(400, 200,), learning_rate='adaptive', 
        learning_rate_init=0.003, max_iter=1000, momentum=0.9, nesterovs_momentum=True, 
        power_t=0.5, random_state=1, shuffle=True, solver='lbfgs', 
        tol=1e-04, validation_fraction=0.1, verbose=True, warm_start=False)

    model.fit(sample_inputs, expected_outputs)

    np.set_printoptions(threshold=np.nan)
    np.save("training_output", model.predict(sample_inputs))

    loaded_expected_outputs = np.load("expected_output.npy")
    loaded_training_outputs = np.load("training_output.npy")

    hit = 0
    miss = 0
    for i in range(len(loaded_expected_outputs)):
        if np.array_equal(loaded_expected_outputs[i], loaded_training_outputs[i]):
            hit += 1
        else:
            miss += 1

    print("Accuracy: " + str(hit / (hit + miss)))
    
    test_dataset = pd.read_csv("test.txt", delimiter = "\n", header = None)
    
    print("Extracting test inputs...")
    
    # This stores the index of a word within the story (starts from 1)
    keyword_id = 0
    keyword_to_id_maps = []
    keyword_to_id_map = {}
    # This stores the index of the current question (starts from 0)
    question_id = 0
    # This stores the index of the current story (starts from 1)
    story_id = 0
    question_id_to_story_id_map = {}
    test_inputs = []
    current_input = [0.0] * input_dimension
    
    hashed_statements = []
    for test_data_point in test_dataset[0]:
        index_content_pair = split_index(test_data_point)
        index = index_content_pair[0]
        content = index_content_pair[1]

        content_tokens = word_tokenize(content)
        pos_tagged_content = nltk.pos_tag(content_tokens)
        # print(pos_tagged_content)
        
        # Convert all verbs in the content into simple present
        for i in range(len(pos_tagged_content)):
            if ("VB" in pos_tagged_content[i][1]):
                pos_tagged_content[i] = (convert_to_present_tense(pos_tagged_content[i][0]), "VB")

        # print(pos_tagged_content)
        
        if (index == "1"):
            # Reset id and map
            keyword_id = 0
            if (len(keyword_to_id_map) > 0):
                keyword_to_id_maps.append(copy.deepcopy(keyword_to_id_map))
                keyword_to_id_map.clear()
            # Reset current input
            current_input = [0.0] * input_dimension
            # Increment story id
            story_id += 1
            # Clear hashed statements for new story
            hashed_statements.clear()
        
        for i in range(len(pos_tagged_content)):
            # Ignore punctuations
            if (pos_tagged_content[i][0] in string.punctuation):
                break
            
            keyword_id += 1
            if ("NNP" in pos_tagged_content[i][1]):
                if (pos_tagged_content[i][0] not in keyword_to_id_map):
                    keyword_to_id_map[pos_tagged_content[i][0]] = keyword_id
            elif ("VB" in pos_tagged_content[i][1]):
                if (pos_tagged_content[i][0] not in keyword_to_id_map):
                    keyword_to_id_map[pos_tagged_content[i][0]] = keyword_id
            elif ("NN" in pos_tagged_content[i][1]):
                if (pos_tagged_content[i][0] not in keyword_to_id_map):
                    keyword_to_id_map[pos_tagged_content[i][0]] = keyword_id
        
        if ("?" in test_data_point):
            question_answer_pair = parse_qna(content)
            question_keywords = question_answer_pair[0]
            
            s_id = 0.0
            for hashed_statement in hashed_statements:
                s_id += 1.0
                current_input[hash_to_input_neuron_id_map[hashed_statement]] = s_id / len(hashed_statements)
            
            for question_keyword in question_keywords:
                hash_value = calculate_hash(question_keyword)
                current_input[hash_to_input_neuron_id_map[hash_value]] = 1.0
                
            test_inputs.append(copy.deepcopy(current_input))
            
            for question_keyword in question_keywords:
                hash_value = calculate_hash(question_keyword)
                current_input[hash_to_input_neuron_id_map[hash_value]] = 0.0
                
            # Remember the story id of this question
            question_id_to_story_id_map[question_id] = story_id
            question_id += 1
            
        else:
            statement_permutations = parse_statement(content)
            
            for statement_permutation in statement_permutations:
                hash_value = (calculate_hash(statement_permutation[0]) + 
                              calculate_hash(statement_permutation[1]) + 
                              calculate_hash(statement_permutation[2])) % 1000000007
                hashed_statements.append(hash_value)
                
    if (len(keyword_to_id_map) > 0):
        keyword_to_id_maps.append(copy.deepcopy(keyword_to_id_map))
        keyword_to_id_map.clear()
    
    print("Classifying test data points...")
    
    np.save("test_output", model.predict(test_inputs))
    np.save("test_output_proba", model.predict_proba(test_inputs))
    test_outputs = np.load("test_output.npy")
    test_outputs_proba = np.load("test_output_proba.npy")
    
    print("Printing normalized outputs...")
    
    normalized_test_outputs_file = open("normalized_test_output.csv", "w")
    
    current_story_id = 0
    current_question_id = 0
    
    print("textID,sortedAnswerList", file = normalized_test_outputs_file)
    for i in range(len(test_outputs)):
        if (question_id_to_story_id_map[i] != current_story_id):
            current_story_id = question_id_to_story_id_map[i]
            current_question_id = 1
        else:
            current_question_id += 1
        
        print(str(current_story_id) + "_" + str(current_question_id) + ",", end = "", 
              flush = True, file = normalized_test_outputs_file)
        
        keyword_ids = []
        for j in range(len(test_outputs[i])):
            if (test_outputs[i][j] == 1):
                current_keyword = output_neuron_id_to_keyword_map[j]
                # print(current_keyword, end = " ", flush = True, file = normalized_test_outputs_file)
                if (current_keyword not in keyword_to_id_maps[current_story_id - 1]):
                    keyword_ids.append(-1)
                else:
                    keyword_ids.append(keyword_to_id_maps[current_story_id - 1][current_keyword])
                    
        keyword_ids.sort()
        is_first = True
        for j in range(len(keyword_ids)):
            if (j == 0):
                print(keyword_ids[j], end = "", flush = True, file = normalized_test_outputs_file)
                if (keyword_ids[j] == -1):
                    break
            else:
                print(" " + str(keyword_ids[j]), end = "", flush = True, file = normalized_test_outputs_file)
        
        # In cases of ambiguity
        if (len(keyword_ids) == 0):
            # Find the output neuron with the largest value
            max_proba_id = 0
            for j in range(len(test_outputs_proba[i])):
                if (test_outputs_proba[i][j] > test_outputs_proba[i][max_proba_id]):
                    max_proba_id = j
            
            current_keyword = output_neuron_id_to_keyword_map[max_proba_id]
            if (current_keyword not in keyword_to_id_maps[current_story_id - 1]):
                print("-1", end = "", flush = True, file = normalized_test_outputs_file)
            else:
                print(keyword_to_id_maps[current_story_id - 1][current_keyword], end = "", 
                      flush = True, file = normalized_test_outputs_file)
        
        print(file = normalized_test_outputs_file)
    
if __name__ == "__main__":
    main()