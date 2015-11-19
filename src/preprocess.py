#coding=utf-8
#!/bin/bash
#Preprocessing Script for Sentiment-Analysis in Twitter

import re

positive_emoticons = [":-)","=)",":)","8)",":]","=]","=>","8-)",":->",":-]",":‖)",":‘)","=*",":-*",":*","<3",";-)",";)",";-]",";]",";->",";>","%-}","B)","B-)","B|","8|",":P","=P",":-D",":D","=D",":-P","=3","xD",":3",":>",":ˆ)",":-3","=>",":->",":-V","=v",":-1","O.o","o.O"]
negative_emoticons = [":‘(",":,(",":‘-(",":,-(",":˜(˜˜",":˜-(",":-(",":(",":[",":-<",":-[","=(",":-@",":-&",":-t",":-z",":<)","}-(",":o",":O",":-o",":-O",":-\\",":/",":-/",":\\"]
negations = ["cant","wont","dont","wouldnt","shouldnt","couldnt","cannot","can not"]
stopwords = ["a","about","above","across","after","again","against","all","almost","alone","along","already","also","although","always","among","an","and","another","any","anybody","anyone","anything","anywhere","are","area","areas","around","as","ask","asked","asking","asks","at","away","b","back","backed","backing","backs","be","became","because","become","becomes","been","before","began","behind","being","beings","best","better","between","big","both","but","by","c","came","can","case","cases","certain","certainly","clear","clearly","come","could","d","did","differ","different","differently","do","does","done","down","downed","downing","downs","during","e","each","early","either","end","ended","ending","ends","enough","even","evenly","ever","every","everybody","everyone","everything","everywhere","f","face","faces","fact","facts","far","felt","few","find","finds","first","for","four","from","full","fully","further","furthered","furthering","furthers","g","gave","general","generally","get","gets","give","given","gives","go","going","good","goods","got","great","greater","greatest","group","grouped","grouping","groups","h","had","has","have","having","he","her","here","herself","high","higher","highest","him","himself","his","how","however","i","if","important","in","interest","interested","interesting","interests","into","is","it","its","itself","j","just","k","keep","keeps","kind","knew","know","known","knows","l","large","largely","last","later","latest","least","less","let","lets","like","likely","long","longer","longest","m","made","make","making","man","many","may","me","member","members","men","might","more","most","mostly","mr","mrs","much","must","my","myself","n","necessary","need","needed","needing","needs","never","new","newer","newest","next","nobody","non","noone","nothing","now","nowhere","number","numbers","o","of","off","often","old","older","oldest","on","once","one","only","open","opened","opening","opens","or","order","ordered","ordering","orders","other","others","our","out","over","p","part","parted","parting","parts","per","perhaps","place","places","point","pointed","pointing","points","possible","present","presented","presenting","presents","problem","problems","put","puts","q","quite","r","rather","really","right","room","rooms","s","said","same","saw","say","says","second","seconds","see","seem","seemed","seeming","seems","sees","several","shall","she","should","show","showed","showing","shows","side","sides","since","small","smaller","smallest","so","some","somebody","someone","something","somewhere","state","states","still","such","sure","t","take","taken","than","that","the","their","them","then","there","therefore","these","they","thing","things","think","thinks","this","those","though","thought","thoughts","three","through","thus","to","today","together","too","took","toward","turn","turned","turning","turns","two","u","under","until","up","upon","us","use","used","uses","v","very","w","want","wanted","wanting","wants","was","way","ways","we","well","wells","went","were","what","when","where","whether","which","while","who","whole","whose","why","will","with","within","without","work","worked","working","works","would","x","y","year","years","yet","you","young","younger","youngest","your","yours","z"]
pattern_url_1 = "^http.*"
pattern_url_2 = "^www.*"
pattern_target = "^@.*"
pattern_negate = "^.*n't"
pattern_hashtag = "^#.*"

replace_url = ""
replace_target = ""
replace_negate = "not"

targetfilepath = "./"
fr = open(targetfilepath+'test_tokens.txt','r')
fw = open(targetfilepath+'preprocessed_test.txt','w+')

tmpL = fr.readlines()

for tweets in tmpL:
	tokenL = tweets.strip('\n').split(' ')
	newtweet = ""
	for token in tokenL:
		newtoken = ""
		if token not in stopwords:
			newtoken = token
			if token in positive_emoticons:
				newtoken = "epositive"
			elif token in negative_emoticons:
				newtoken = "enegative"
			elif token.lower() in negations:
				newtoken = "not"
			else:
				if re.match(pattern_url_1,token,re.I):
					newtoken = re.sub(pattern_url_1,replace_url,token,re.I)
				if re.match(pattern_url_2,token,re.I):
					newtoken = re.sub(pattern_url_2,replace_url,token,re.I)
				if re.match(pattern_target,token,re.I):
					newtoken = re.sub(pattern_target,replace_target,token,re.I)
				if re.match(pattern_negate,token,re.I):
					newtoken = re.sub(pattern_negate,replace_negate,token,re.I)
				if re.match(pattern_hashtag,token,re.I):
					newtoken = token[1:]
		if len(newtoken):
			if len(newtweet):
				newtweet += " "+newtoken
			else:
				newtweet += newtoken

	newtweet += '\n'
	fw.write(newtweet)

fr.close()
fw.close()
