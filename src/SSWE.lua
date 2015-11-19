
-- hardcoded window size as 2

-- require('mobdebug').start();
require("sys")
require("nn")

local SSWE = torch.class ("SSWE")

----------------------------------- constructor ---------------------------
function SSWE:__init()
    self.neg_samples = 5;
    self.criterion = nn.MarginCriterion()
    self.senti_criterion = nn.MarginCriterion()
    self.com_criterion = nn.ParallelCriterion();
    self.com_criterion : add (self.criterion, 0.6) : add (self.senti_criterion, 0.4);
    self.epochs = 5;
    self.dim = 50;
    self.window_size = 1;
    
    self.contexts = torch.IntTensor(4+self.neg_samples) 
    self.labels = torch.zeros(4+self.neg_samples); 
    self.labels[1] = 1; self.labels[2] = 1; self.labels[3] = 1; self.labels[4] = 1 -- first label is always pos sample           -- refers to both sides of the target word
    
    self.senti_contexts = torch.IntTensor (4);
    self.senti_labels = torch.Tensor (4);
    self.lr = .01;

    self.vocab = {}
    self.index2word = {}
    self.word = torch.IntTensor (1) 
end

---------------------------------- building vocab --------------------------
function SSWE:build_vocab(corpus_file_name)
    local f = io.open (corpus_file_name, "r")
    local i = 1
    for line in f : lines () do
        for _, word in ipairs (self:split (line)) do
            if self.vocab[word] == nil then
                self.vocab[word] = i
                self.index2word[i] = word
                i = i + 1
            end
        end
    end
    f : close ();

    self.vocab_size = #self.index2word
end

------------------------------------ split on separator -------------------------
function SSWE:split(input, sep)
    if sep == nil then
        sep = "%s"
    end
    local t = {}; local i = 1
    for str in string.gmatch(input, "([^"..sep.."]+)") do
        t[i] = str; i = i + 1
    end
    return t
end

------------------------------------ training -----------------------------------
function SSWE:train_model (corpus_file_name, sentiment_file_name)

    -- Describing the model
    self.word_vecs = nn.LookupTable(self.vocab_size, self.dim) -- word embeddings
    self.context_vecs = nn.LookupTable(self.vocab_size, self.dim) -- context embeddings

    self.senti_word_vecs = self.word_vecs : clone ('weight', 'bias', 'gradWeight', 'gradBias')
    self.senti_context_vecs = self.context_vecs : clone ('weight', 'bias', 'gradWeight', 'gradBias')
    
    -- semantic model
    self.model = nn.Sequential()
    self.model:add(nn.ParallelTable())
    self.model.modules[1]:add(self.context_vecs)
    self.model.modules[1]:add(self.word_vecs)
    self.model:add (nn.MM(false, true)); -- dot prod and sigmoid to get probabilities
    self.model:add (nn.HardTanh ());

    -- self.trainer = nn.StochasticGradient(self.model,self.criterion)
    -- self.trainer.learningRate = self.lr
    -- self.trainer.maxIteration = self.epochs


    -- sentiment model
    self.senti_model = nn.Sequential();
    self.senti_model : add (nn.ParallelTable())
    self.senti_model.modules[1]:add(self.senti_context_vecs)
    self.senti_model.modules[1]:add(self.senti_word_vecs)
    self.senti_model : add (nn.MM(false, true));
    self.senti_model : add (nn.HardTanh ());
    
    -- self.senti_trainer = nn.StochasticGradient(self.senti_model,self.senti_criterion)
    -- self.senti_trainer.learningRate = self.lr
    -- self.senti_trainer.maxIteration = self.epochs


    -- complete model
    -- self.com_model = nn.Sequential ();
    self.com_model = nn.ParallelTable ();
    self.com_model : add (self.model);
    self.com_model : add (self.senti_model);

    -- complete trainer
    self.com_trainer = nn.StochasticGradient (self.com_model, self.com_criterion);
    self.com_trainer.verbose = false;
    self.com_trainer.learningRate = self.lr;
    self.com_trainer.maxIteration = self.epochs;

    -- Now beginning the training
    print ("training...")
    local c = 0;
    local start = sys.clock ()
    f = io.open (corpus_file_name, "r")
    f1 = io.open (sentiment_file_name, "r")
    local z = 1;
    for line in f : lines () do
        if z % 100 == 1 then
            print(string.format ("Training line %d...%.2f ?seconds gone", z, sys.clock () - start));
        end
        z = z + 1;

        self.sentiment = tonumber(f1:read ())
        sentence = self : split (line)
        for i, word in ipairs (sentence) do
            self.word[1] = self.vocab[word]
            
            -- find contexts and negative-contexts
            if sentence[i - 1] ~= nil and sentence[i + 1] ~= nil 
                and sentence[i - 2] ~= nil and sentence[i + 2] ~= nil then
                self.contexts[1] = self.vocab[sentence[i - 1]];
                self.contexts[2] = self.vocab[sentence[i + 1]];
                self.contexts[3] = self.vocab[sentence[i - 2]];
                self.contexts[4] = self.vocab[sentence[i + 2]];
                
                self.senti_contexts[1] = self.vocab[sentence[i - 1]];
                self.senti_contexts[2] = self.vocab[sentence[i + 1]];
                self.senti_contexts[3] = self.vocab[sentence[i - 2]];
                self.senti_contexts[4] = self.vocab[sentence[i + 2]];
                self.senti_labels[1] = self.sentiment;
                self.senti_labels[2] = self.sentiment;
                self.senti_labels[3] = self.sentiment;
                self.senti_labels[4] = self.sentiment;

                -- now for the negative contexts
                local j = 0;
                while j < self.neg_samples do
                    neg_context = self.index2word[torch.random (self.vocab_size)]

                    if neg_context ~= sentence[i - 1] 
                        and neg_context ~= sentence[i + 1]
                        and neg_context ~= sentence[i - 2] 
                        and neg_context ~= sentence[i + 2] 
                        and neg_context ~= word then
                        self.contexts[j + 5] = self.vocab[neg_context]
                        j = j + 1
                    end
                end

                -- training
                data = {}
                function data:size () return 1 end
                data[1] = {
                  {{self.contexts, self.word}, {self.senti_contexts, self.word}},
                  {self.labels, self.senti_labels}
                            };

                self.com_trainer:train(data);
            end
        end
    end
    f:close();
end

-------------------------------------------------------------------------------------------------------
---------------------------------Module for testing ---------------------------------------------------

---------------------------------- Row-normalize a matrix ---------------------------------------------
function SSWE:normalize(m)
    m_norm = torch.zeros(m:size())
    for i = 1, m:size(1) do
        m_norm[i] = m[i] / torch.norm(m[i])
    end
    return m_norm
end

-------------------------------- Get similar words -------------------------------------------------
-- Return the k-nearest words to a word or a vector based on cosine similarity
-- w can be a string such as "king" or a vector for ("king" - "queen" + "man")
function SSWE:get_sim_words(w, k)
    if self.word_vecs_norm == nil then
        self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
    end
    if type(w) == "string" then
        if self.vocab[w] == nil then
       print("'"..w.."' does not exist in vocabulary.")
       return nil
    else
            w = self.word_vecs_norm[self.vocab[w]]
    end
    end
    local sim = torch.mv(self.word_vecs_norm, w)
    sim, idx = torch.sort(-sim)
    local r = {}
    for i = 1, k do
        r[i] = {self.index2word[idx[i]], -sim[i]}
    end
    return r
end

-- print similar words
function SSWE:print_sim_words(words, k)
    for i = 1, #words do
        r = self:get_sim_words(words[i], k)
    if r ~= nil then
        print("-------"..words[i].."-------")
        for j = 1, k do
            print(string.format("%s, %.4f", r[j][1], r[j][2]))
        end
    end
    end
end