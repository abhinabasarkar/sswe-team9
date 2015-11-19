-- require('mobdebug').start();
require ("io")
require("os")
require("paths")
require("torch")
dofile ("SSWE.lua")

corpus_file = "superCleaned_last_4.txt";
sentiment_file = "labels_last_4.txt";

-- model
m = SSWE ();

m : build_vocab (corpus_file)
print (string.format ("%d words in the vocab", m.vocab_size))
m : train_model (corpus_file, sentiment_file)
word_vector = m.word_vecs.weight

torch.save ('small_word_embeddings.dat', word_vector);
torch.save ('small_vocab.dat', m.vocab);
torch.save ('small_index2word.dat', m.index2word);

-- print(word_vector)

print (string.format ("%d words in the vocab", m.vocab_size))

f = io.open ('small_vectors_last_4.bin', 'w');
io.output (f);

for i = 1, m.vocab_size do
    io.write (m.index2word[i] .. " ")
    for j = 1, 49 do
        io.write (word_vector[i][j] .. " ")
    end
    io.write(word_vector[i][50] .. "\n")
end

-- print (torch.load ('vocab.dat'));

-- m:print_sim_words({"video","microsoft"},5)
io.close (f);
-- torch.save ("model_try1.dat", m);