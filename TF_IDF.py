


import math

from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords

text_str = '''
Those Who Are Resilient Stay In The Game Longer
“On the mountains of truth you can never climb in vain: either you will reach a point higher up today, or you will be training your powers so that you will be able to climb higher tomorrow.” — Friedrich Nietzsche
Challenges and setbacks are not meant to defeat you, but promote you. However, I realise after many years of defeats, it can crush your spirit and it is easier to give up than risk further setbacks and disappointments. Have you experienced this before? To be honest, I don’t have the answers. I can’t tell you what the right course of action is; only you will know. However, it’s important not to be discouraged by failure when pursuing a goal or a dream, since failure itself means different things to different people. To a person with a Fixed Mindset failure is a blow to their self-esteem, yet to a person with a Growth Mindset, it’s an opportunity to improve and find new ways to overcome their obstacles. Same failure, yet different responses. Who is right and who is wrong? Neither. Each person has a different mindset that decides their outcome. Those who are resilient stay in the game longer and draw on their inner means to succeed.
I’ve coached mummy and mom clients who gave up after many years toiling away at their respective goal or dream. It was at that point their biggest breakthrough came. Perhaps all those years of perseverance finally paid off. It was the 19th Century’s minister Henry Ward Beecher who once said: “One’s best success comes after their greatest disappointments.” No one knows what the future holds, so your only guide is whether you can endure repeated defeats and disappointments and still pursue your dream. Consider the advice from the American academic and psychologist Angela Duckworth who writes in Grit: The Power of Passion and Perseverance: “Many of us, it seems, quit what we start far too early and far too often. Even more than the effort a gritty person puts in on a single day, what matters is that they wake up the next day, and the next, ready to get on that treadmill and keep going.”
I know one thing for certain: don’t settle for less than what you’re capable of, but strive for something bigger. Some of you reading this might identify with this message because it resonates with you on a deeper level. For others, at the end of their tether the message might be nothing more than a trivial pep talk. What I wish to convey irrespective of where you are in your journey is: NEVER settle for less. If you settle for less, you will receive less than you deserve and convince yourself you are justified to receive it.
“Two people on a precipice over Yosemite Valley” by Nathan Shipps on Unsplash
Develop A Powerful Vision Of What You Want
“Your problem is to bridge the gap which exists between where you are now and the goal you intend to reach.” — Earl Nightingale
I recall a passage my father often used growing up in 1990s: “Don’t tell me your problems unless you’ve spent weeks trying to solve them yourself.” That advice has echoed in my mind for decades and became my motivator. Don’t leave it to other people or outside circumstances to motivate you because you will be let down every time. It must come from within you. Gnaw away at your problems until you solve them or find a solution. Problems are not stop signs, they are advising you that more work is required to overcome them. Most times, problems help you gain a skill or develop the resources to succeed later. So embrace your challenges and develop the grit to push past them instead of retreat in resignation. Where are you settling in your life right now? Could you be you playing for bigger stakes than you are? Are you willing to play bigger even if it means repeated failures and setbacks? You should ask yourself these questions to decide whether you’re willing to put yourself on the line or settle for less. And that’s fine if you’re content to receive less, as long as you’re not regretful later.
If you have not achieved the success you deserve and are considering giving up, will you regret it in a few years or decades from now? Only you can answer that, but you should carve out time to discover your motivation for pursuing your goals. It’s a fact, if you don’t know what you want you’ll get what life hands you and it may not be in your best interest, affirms author Larry Weidel: “Winners know that if you don’t figure out what you want, you’ll get whatever life hands you.” The key is to develop a powerful vision of what you want and hold that image in your mind. Nurture it daily and give it life by taking purposeful action towards it.
Vision + desire + dedication + patience + daily action leads to astonishing success. Are you willing to commit to this way of life or jump ship at the first sign of failure? I’m amused when I read questions written by millennials on Quora who ask how they can become rich and famous or the next Elon Musk. Success is a fickle and long game with highs and lows. Similarly, there are no assurances even if you’re an overnight sensation, to sustain it for long, particularly if you don’t have the mental and emotional means to endure it. This means you must rely on the one true constant in your favour: your personal development. The more you grow, the more you gain in terms of financial resources, status, success — simple. If you leave it to outside conditions to dictate your circumstances, you are rolling the dice on your future.
So become intentional on what you want out of life. Commit to it. Nurture your dreams. Focus on your development and if you want to give up, know what’s involved before you take the plunge. Because I assure you, someone out there right now is working harder than you, reading more books, sleeping less and sacrificing all they have to realise their dreams and it may contest with yours. Don’t leave your dreams to chance.
'''


def _create_frequency_table(text_string) -> dict:
    """
    we create a dictionary for the word frequency table.
    """
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


'''
{'\nThose Who Are ': {'resili': 1, 'stay': 1, 'game': 1, 'longer': 1, '“': 1, 'mountain': 1}, 'However, I real': {'howev': 1, ',': 2, 'realis': 1, 'mani': 1, 'year': 1}, 'Have you experi': {'experienc': 1, 'thi': 1, 'befor': 1, '?': 1}, 'To be honest, I': {'honest': 1, ',': 1, '’': 1, 'answer': 1, '.': 1}, 'I can’t tell yo': {'’': 1, 'tell': 1, 'right': 1, 'cours': 1, 'action': 1, ';': 1, 'onli': 1, 'know': 1, '.': 1}...}

'''


def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix



'''
{'\nThose Who Are ': {'resili': 0.03225806451612903, 'stay': 0.03225806451612903, 'game': 0.03225806451612903, 'longer': 0.03225806451612903, '“': 0.03225806451612903, 'mountain': 0.03225806451612903}, 'However, I real': {'howev': 0.07142857142857142, ',': 0.14285714285714285, 'realis': 0.07142857142857142, 'mani': 0.07142857142857142, 'year': 0.07142857142857142}, 'Have you experi': {'experienc': 0.25, 'thi': 0.25, 'befor': 0.25, '?': 0.25}, 'To be honest, I': {'honest': 0.2, ',': 0.2, '’': 0.2, 'answer': 0.2, '.': 0.2}, 'I can’t tell yo': {'’': 0.1111111111111111, 'tell': 0.1111111111111111, 'right': 0.1111111111111111, 'cours': 0.1111111111111111, 'action': 0.1111111111111111, ';': 0.1111111111111111, 'onli': 0.1111111111111111, 'know': 0.1111111111111111, '.': 0.1111111111111111}}
'''


def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table



'''

{'resili': 2, 'stay': 2, 'game': 3, 'longer': 2, '“': 5, 'mountain': 1, 'truth': 1, 'never': 2, 'climb': 1, 'vain': 1, ':': 8, 'either': 1, 'reach': 1, 'point': 2, 'higher': 1, 'today': 1, ',': 22, 'train': 1, 'power': 4, 'abl': 1, 'tomorrow.': 1, '”': 5, '—': 3, 'friedrich': 1, 'nietzsch': 1, 'challeng': 2, 'setback': 2, 'meant': 1, 'defeat': 3, 'promot': 1, '.': 45, 'howev': 2, 'realis': 2, 'mani': 3, 'year': 4, 'crush': 1, 'spirit': 1, 'easier': 1, 'give': 4, 'risk': 1}


'''

def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


'''
{'\nThose Who Are ': {'resili': 1.414973347970818, 'stay': 1.414973347970818, 'game': 1.2388820889151366, 'longer': 1.414973347970818, '“': 1.0170333392987803, 'mountain': 1.7160033436347992}, 'However, I real': {'howev': 1.414973347970818, ',': 0.37358066281259295, 'realis': 1.414973347970818, 'mani': 1.2388820889151366, 'year': 1.1139433523068367}, 'Have you experi': {'experienc': 1.7160033436347992, 'thi': 1.1139433523068367, 'befor': 1.414973347970818, '?': 0.9378520932511555}, 'To be honest, I': {'honest': 1.7160033436347992, ',': 0.37358066281259295, '’': 0.5118833609788743, 'answer': 1.414973347970818, '.': 0.06279082985945544}, 'I can’t tell yo': {'’': 0.5118833609788743, 'tell': 1.414973347970818, 'right': 1.1139433523068367, 'cours': 1.7160033436347992, 'action': 1.2388820889151366, ';': 1.7160033436347992, 'onli': 1.2388820889151366, 'know': 1.0170333392987803, '.': 0.06279082985945544}}


'''

def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix





'''

{'\nThose Who Are ': {'resili': 0.04564430154744574, 'stay': 0.04564430154744574, 'game': 0.03996393835210118, 'longer': 0.04564430154744574, '“': 0.0328075270741542, 'mountain': 0.05535494656886449}, 'However, I real': {'howev': 0.10106952485505842, ',': 0.053368666116084706, 'realis': 0.10106952485505842, 'mani': 0.08849157777965261, 'year': 0.07956738230763119}, 'Have you experi': {'experienc': 0.4290008359086998, 'thi': 0.2784858380767092, 'befor': 0.3537433369927045, '?': 0.23446302331278887}, 'To be honest, I': {'honest': 0.34320066872695987, ',': 0.07471613256251859, '’': 0.10237667219577487, 'answer': 0.2829946695941636, '.': 0.01255816597189109}, 'I can’t tell yo': {'’': 0.0568759289976527, 'tell': 0.15721926088564644, 'right': 0.12377148358964851, 'cours': 0.19066703818164435, 'action': 0.13765356543501517, ';': 0.19066703818164435, 'onli': 0.13765356543501517, 'know': 0.11300370436653114, '.': 0.006976758873272827}}
'''


def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF

    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


'''

{'\nThose Who Are ': 0.049494684794344025, 'However, I real': 0.09203831532832171, 'Have you experi': 0.3239232585727256, 'To be honest, I': 0.16316926181026162, 'I can’t tell yo': 0.12383203821623005}

'''


def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average


'''   0.15611302409372044 '''

def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def run_summarization(text):

   
    # 1 Sentence Tokenize
    sentences = sent_tokenize(text)
    total_documents = len(sentences)
    #print(sentences)

    # 2 Create the Frequency matrix of the words in each sentence.
    freq_matrix = _create_frequency_matrix(sentences)
    #print(freq_matrix)

    '''
    Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
    '''
    # 3 Calculate TermFrequency and generate a matrix
    tf_matrix = _create_tf_matrix(freq_matrix)
    #print(tf_matrix)

    # 4 creating table for documents per words
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    #print(count_doc_per_words)

    '''
    Inverse document frequency (IDF) is how unique or rare a word is.
    '''
    # 5 Calculate IDF and generate a matrix
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    #print(idf_matrix)

    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    #print(tf_idf_matrix)

    # 7 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(tf_idf_matrix)
    #print(sentence_scores)

    # 8 Find the threshold
    threshold = _find_average_score(sentence_scores)
    #print(threshold)

    # 9 Important Algorithm: Generate the summary 
   
    summary = _generate_summary(sentences, sentence_scores,   1.3 * threshold)
    return summary


if __name__ == '__main__':
    result = run_summarization(text_str)
    print(result)
