import numpy as np
from subprocess import call

#this readies the data for svm
data_folder_path = '../data/'
file_names = ['test']
word_embedding_size = 50

# read the vectors.bin file and create a map
vec_dict = {}
# vec_file = open ('vectors.bin', 'r')

# create vectors and counts
for f in file_names:
    with open (data_folder_path + f, 'r') as fp:
        for i, line in enumerate (fp):
            parts = line.split ()
            word = parts[0]

            # word is already present in the dictionary
            if word in vec_dict:
                com_vec = vec_dict[word]
                vector = []
                for i in range (1, word_embedding_size + 1):
                    try:
                        vector.append (float (parts[i]))
                    except Exception:
                        vector.append (0.0)
                com_vec[0] = com_vec[0] + np.array (vector)
                com_vec[1] += 1
                vec_dict[word] = com_vec
            
            # word is not present in the dictionary
            else:
                vector = []
                for i in range (1, word_embedding_size + 1):
                    try:
                        vector.append (float (parts[i]))
                    except Exception:
                        vector.append (0.0)
                vec_dict[word] = [np.array (vector), 1]

# calculating the average
for word in vec_dict:
    vector, cnt = vec_dict[word]
    vector = vector / cnt;
    vec_dict[word] = vector

# now we read the train data
with open (data_folder_path + 'train.tsv', 'r') as fp:
    with open (data_folder_path + 'training_data_svm', 'w') as fpw:
        for line in fp:
            final_vec = np.zeros (word_embedding_size)
            [tweet, label] = line.split ("\t")
            tweet_parts = tweet.split ()
            c = 0
            for word in tweet_parts:
                c += 1
                if word in vec_dict:
                    final_vec = final_vec + vec_dict[word]

            final_vec = final_vec / c
            write_line = label.strip ("\n").strip ()
            for i in range (word_embedding_size):
                write_line += " "
                write_line += str (i + 1) + ":"
                write_line += str (final_vec[i])
            write_line += "\n"
            fpw.write (write_line)


# now for the test data
with open (data_folder_path + 'test.tsv', 'r') as fp:
    with open (data_folder_path + 'test_data_svm', 'w') as fpw:
        for line in fp:
            final_vec = np.zeros (word_embedding_size)
            [tweet, label] = line.split ("\t")
            tweet_parts = tweet.split ()
            c = 0
            for word in tweet_parts:
                c += 1
                if word in vec_dict:
                    final_vec = final_vec + vec_dict[word]
            
            final_vec = final_vec / c
            write_line = label.strip ("\n").strip ()
            for i in range (word_embedding_size):
                write_line += " "
                write_line += str (i + 1) + ":"
                write_line += str (final_vec[i])
            write_line += "\n"
            fpw.write (write_line)


# call(["python", "easy.py", "training_data_svm", "test_data_svm"])
print ""
