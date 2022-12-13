from EvolutionaryModelTest import GAHelpers

helpers = GAHelpers()
threshold = 0.01
results = []

def get_summary_score(article, abstract):
    article = helpers.filter_sentence(article, "article")
    summary = helpers.summarize(dictionary, article, threshold)
    score = helpers.score_summary(summary, abstract)
    return score

for POP_SIZE, GEN_N in [(50, 10)]:
    for SAMPLE_SIZE in [250]:
        for VOCAB_SIZE in [1000]:

            articles, abstracts = helpers.read_articles(SAMPLE_SIZE)
            vocab = helpers.read_list()[:VOCAB_SIZE]
            weights = helpers.read_list('./vocab_files/new_vocab_g' + str(GEN_N) + '_p' + str(POP_SIZE) + '_a' + helpers.get_sample_size_str(SAMPLE_SIZE) + '_v' + helpers.get_vocab_size_str(VOCAB_SIZE))
            dictionary = helpers.update_weights(vocab, weights)

            best_score = 0

            for article, abstract in zip(articles, abstracts):
                score = get_summary_score(article, abstract)
                if score > best_score:
                    best_score = score

            rec = {'pop': POP_SIZE, 'sample': SAMPLE_SIZE, 'vocab': VOCAB_SIZE, 'result': best_score}
            results.append(rec)
            print(rec)

print(results)
