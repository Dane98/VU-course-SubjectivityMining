import logging
import sys

from tasks import vua_format as vf
from ml_pipeline import utils, cnn, preprocessing, pipeline_with_lexicon
from ml_pipeline import pipelines
from ml_pipeline.cnn import CNN, evaluate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
#handler = logging.FileHandler('experiment.log')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def run(task_name, data_dir, pipeline_name, print_predictions):
    logger.info('>> Running {} experiment'.format(task_name))
    tsk = task(task_name)
    logger.info('>> Loading data...')
    tsk.load(data_dir)
    logger.info('>> retrieving train/data instances...')
    train_X, train_y, test_X, test_y = utils.get_instances(
        tsk, split_train_dev=False)
    test_X_ref = test_X

    if pipeline_name.startswith('cnn'):
        pipe = cnn(pipeline_name)
        train_X, train_y, test_X, test_y = pipe.encode(
            train_X, train_y, test_X, test_y)
        logger.info('>> testing...')
    else:
        pipe = pipeline(pipeline_name)

    logger.info('>> training pipeline ' + pipeline_name)
    pipe.fit(train_X, train_y)
    if pipeline_name == 'naive_bayes_counts_lex':
        logger.info(
            "   -- Found {} tokens in lexicon".format(pipe.tokens_from_lexicon))

    logger.info('>> testing...')
    sys_y = pipe.predict(test_X)

    logger.info('>> evaluation...')
    logger.info(utils.eval(test_y, sys_y))
    utils.important_features_per_class(pipe.named_steps.frm, pipe.named_steps.clf)   # Added # (frm, clf can be found in pipelines.py)

    
    if print_predictions:
        logger.info('>> predictions')
        #utils.print_all_predictions(test_X_ref, test_y, sys_y, logger)

        #file_name = data_dir.split("/")[-2]+'_'+'prediction'+'.txt'
        #file_name = 'olid_'+'prediction'+'.txt'
        file_name = 'hasoc_'+'prediction'+'.txt'
        file = open('./'+file_name,mode="w",encoding='utf-8')
        for i in range(0, len(sys_y)):
            to_print = "{}\t{}\t{}\n".format(sys_y[i], test_y.values[i], test_X_ref[i])
            file.write(to_print)
        file.close()

    #if print_best:
        #logger.info('>> optimizing')
        #utils.report(grid_search.cv_results_, n_top=10)

def task(name):
    if name == 'vua_format':
        return vf.VuaFormat()
    else:
        raise ValueError(
            "task name is unknown. You can add a custom task in 'tasks'")


def cnn(name):
    if name == 'cnn_raw':
        return CNN()
    elif name == 'cnn_prep':
        return CNN(preprocessing.std_prep())
    else:
        raise ValueError("pipeline name is unknown.")


def pipeline(name):
    if name == 'naive_bayes_counts':
        return pipelines.naive_bayes_counts()
    elif name == 'naive_bayes_tfidf':
        return pipelines.naive_bayes_tfidf()
    elif name == 'naive_bayes_counts_lex':
        return pipeline_with_lexicon.naive_bayes_counts_lex()
    elif name == 'svm_libsvc_counts':
        return pipelines.svm_libsvc_counts()
    elif name == 'svm_libsvc_tfidf':
        return pipelines.svm_libsvc_tfidf()
    elif name == 'svm_libsvc_embed':
        return pipelines.svm_libsvc_embed()
    elif name == 'svm_sigmoid_embed':
        return pipelines.svm_sigmoid_embed()
    # custom
    elif name == 'naive_bayes_counts_bigram':
        return pipelines.naive_bayes_counts_bigram()
    #elif name == 'naive_bayes_counts_lex':
        #return pipelines.naive_bayes_counts_lex()
    else:
        raise ValueError(
            "pipeline name is unknown. You can add a custom pipeline in 'pipelines'")
