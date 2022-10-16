from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from ml_pipeline import preprocessing, representation
from sklearn import svm

class CombinedFeaturesPipeline:
    def __init__(self, prep1, repr1, prep2, repr2, classifier):
        combined_features = FeatureUnion([
                ('token_features', Pipeline([('prep1', prep1), ('repr1', repr1)])),
                ('lexicon_features', Pipeline([('prep2', prep2), ('repr2', repr2)]))])
        self.pipeline = Pipeline([('features', combined_features), ('clf', classifier)])
        self.tokens_from_lexicon = 0

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        self.tokens_from_lexicon = self.pipeline.steps[0][1].transformer_list[1][1].steps[0][1].tokens_from_lexicon

    def predict(self, X):
        return self.pipeline.predict(X)


def naive_bayes_counts_lex(lex_name):
    return CombinedFeaturesPipeline(preprocessing.std_prep(), representation.count_vectorizer({'min_df': 1}),
                                    preprocessing.lex_prep(lex_name), representation.count_vectorizer({'min_df': 1}),
                                    MultinomialNB())

#custom
def svm_libsvc_counts_lex(lex_name):
    return CombinedFeaturesPipeline(preprocessing.std_prep(), representation.count_vectorizer({'min_df': 1}),
                                    preprocessing.lex_prep(lex_name), representation.count_vectorizer({'min_df': 1}),
                                    svm.LinearSVC(max_iter=10000, dual=False, C=0.1))                                    


def svm_libsvc_tfidf_lex(lex_name):
    return CombinedFeaturesPipeline(preprocessing.std_prep(), representation.tfidf_vectorizer({'analyzer':'char','min_df': 10,'ngram_range': (1,6)}),
                                    preprocessing.lex_prep(lex_name), representation.tfidf_vectorizer({'analyzer':'char','min_df': 10,'ngram_range': (1,6)}),
                                    svm.LinearSVC(max_iter=10000, dual=False, C=0.1))              