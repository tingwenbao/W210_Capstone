#!/usr/bin/env python3
'''
Builds the machine learning model using capstone project data
Author: Sombiri Enwemeka
'''

from db_crud import DB_CRUD
from db_object import DB_Object
from ml_test_params import *
import numpy as np
from bson.objectid import ObjectId
from pickle import dumps as pdumps
from pickle import dump as pdump
from pickle import load as pload
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from demo_multipliers import get_multiplier
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

PROD_COMO = []
T_TYPE = 'product'

PEOPLE_DB = None
PRODUCTS_DB = None
INGREDIENTS_DB = None
MODEL_DB = None


def build_product_model(host, port, **kwargs):
    prod_model_data = 'prod_model_data.pickle'
    print("Loading products from database:")
    prod_filt = {'comodegenic': {'$type': 'int'}}  # Only return entries with comodegenic score
    prod_prjctn = {
        'ingredient_list': True,
        'comodegenic': True}
    db_objects = PRODUCTS_DB.read(prod_filt, projection=prod_prjctn)
    products = [DB_Object.build_from_dict(p) for p in db_objects]

    # The tfidf_vect will ignore the following words
    stop_words = [
        '',
        'water',
        'glycerin',
        'titanium dioxide',
        'iron oxides',
        'beeswax',
        'methylparaben',
        'propylparaben',
        'propylene glycol',
        'panthenol',
        'mica']

    # Tokenizer for product ingredient lists
    def get_prod_ings_as_list(product):
        '''
        Queries the ingredients DB for a given product's ingredient list
        and returns the ingredient list as a list of ingredient strings
        Note: The DB query is performed once using all ingredient object
        IDs simultaneously.
        '''
        fltr = {'_id': {'$in': product.get('ingredient_list', [])}}
        ing_prjctn = {'_id': False, 'ingredient_name': True}
        db_objects = INGREDIENTS_DB.read(fltr, projection=ing_prjctn)
        return [DB_Object.build_from_dict(i).get('ingredient_name', '') for i in db_objects]

    print('Vectorizing product ingredient lists')
    tfidf_vect = TfidfVectorizer(
        tokenizer=get_prod_ings_as_list,
        lowercase=False,
        stop_words=stop_words)
    X = tfidf_vect.fit_transform(products)
    y = [p['comodegenic'] for p in products]

    print('Storing vectorized data and training labels')
    # Flatten CSR sparse matrix to strings
    model = {
        'X': X,
        'y': y
    }

    print("Saving model data to disk for next time")
    # Insert the model into the model database
    MODEL_DB.create_file(pdumps(model, protocol=2), filename="ml_product_data")
    # Save model data to disk
    with open(prod_model_data, "wb") as pickle_out:
        pdump(model, pickle_out)
    print('[SUCCESS] Product model data post-processed and stored')


def get_ingredient_vocabulary(host, port, **kwargs):
    ''' Returns the set of all unique ingredient names including synonyms
    '''
    # Build list of all ingredient names
    ing_fltr = {}  # Get all ingredients
    ing_prjctn = {
        '_id': False,
        'ingredient_name': True,
        'synonym_list': True}
    db_objects = INGREDIENTS_DB.read(ing_fltr, projection=ing_prjctn)
    ingredients = [DB_Object.build_from_dict(i) for i in db_objects]
    ret = set()
    for ingredient in ingredients:
        ret.update([ingredient.get('ingredient_name', '')])
        for synonym in ingredient.get('synonym_list', []):
            ret.update([ingredient.get('ingredient_name', '')])
    return ret


def set_tokenizer_type(t_type):
    global T_TYPE
    if t_type not in ['product', 'ingredient']:
        raise RuntimeError(
            "The input tokenizer type '"
            + str(t_type)
            + "' was invalid")
    else:
        T_TYPE = t_type
    pass


# Tokenizer for ingredient lists
def get_ingredients_as_list(p_list_or_i):
    '''
    Queries the products and ingredients DBs for ingredients contained within
    the products given by the input list of Object_Ids. Changing the tokenizer
    type variable 'T_TYPE' to ingredient causes this function to expect ObjectIds
    referring to infredients as input.
    Note: The each DB query is performed once using all object
    IDs simultaneously. This function performs no more than 2 queries when run.
    '''
    global PROD_COMO

    if not p_list_or_i:
        return []
    elif type(p_list_or_i) is str or type(p_list_or_i) is ObjectId:
        # Query a single ObjectId
        prod_fltr = {'_id': p_list_or_i}
    else:
        # Build list of ingredient ObjectIds contained in the product list
        prod_fltr = {'_id': {'$in': p_list_or_i}}

    if T_TYPE == 'product':
        prod_prjctn = {
            '_id': False,
            'ingredient_list': True,
            'comodegenic': True}
        db_objects = PRODUCTS_DB.read(prod_fltr, projection=prod_prjctn)

        # Get ObjectIds from all product ingredients
        ing_list = set()  # Using set eliminates duplicate values
        for i in db_objects:
            ing = DB_Object.build_from_dict(i)
            ing_list.update(ing.get('ingredient_list', ''))
            PROD_COMO.append(ing.get('comodegenic', 0))  # Create column of comodegenic scores

        # Build list of all ingredient names
        ing_fltr = {'_id': {'$in': list(ing_list)}}
        ing_prjctn = {'_id': False, 'ingredient_name': True}
        db_objects = INGREDIENTS_DB.read(ing_fltr, projection=ing_prjctn)
        return [DB_Object.build_from_dict(i).get('ingredient_name', '') for i in db_objects]
    else:
        # Return the ingredient name
        ing_fltr = {'_id': p_list_or_i}
        ing_prjctn = {'_id': False, 'ingredient_name': True}
        db_objects = INGREDIENTS_DB.read(ing_fltr, projection=ing_prjctn)
        return [DB_Object.build_from_dict(i).get('ingredient_name', '') for i in db_objects]


def build_people_model(host, port, **kwargs):
    global PROD_COMO
    ppl_model_data = 'ppl_model_data.pickle'
    batch_size = kwargs.get('batch_size', 10000)
    vocabulary = get_ingredient_vocabulary(host, port)

    # The tfidf_vect will ignore the following words
    stop_words = [
        '',
        'water',
        'glycerin',
        'titanium dioxide',
        'iron oxides',
        'beeswax',
        'methylparaben',
        'propylparaben',
        'propylene glycol',
        'panthenol',
        'mica']

    # Create vectorizers
    d_vect = DictVectorizer(sparse=False)
    tfidf_vect = TfidfVectorizer(
        tokenizer=get_ingredients_as_list,
        lowercase=False,
        stop_words=stop_words,
        vocabulary=vocabulary)

    print("Loading people from database, batch_size:", str(batch_size))
    ppl_filt = {}
    ppl_prjctn = {
        '_id': False,
        'race': True,
        'birth_sex': True,
        'age': True,
        'acne': True,
        'skin': True,
        'acne_products': True}  # Don't include any PII
    db_objects = PEOPLE_DB.read(ppl_filt, projection=ppl_prjctn)

    y, demo_mult = [], []
    batch_num, pulled = 0, 0
    X = None

    # Work in batches to build dataset
    while pulled <= db_objects.count(with_limit_and_skip=True):
        # Initialize
        X_demo_lst, X_prod_lst = [], []
        people = []

        print('Parsing batch:', batch_num)

        try:
            # Build a batch
            for i in range(batch_size):
                people.append(DB_Object.build_from_dict(db_objects.next()))
                pulled += 1
        except StopIteration:
        # End of available data
            break

        # Extract features
        for person in people:
            # Create new entry for each product
            # Note: Model is only applicable to entries with products
            for product_id in person.pop('acne_products'):
                # Pull product ingredients info
                X_prod_lst.append([product_id])

                # Pull demographic info
                X_demo_lst.append(person)

                # Generate demographic multiplier
                mult = get_multiplier(person)
                demo_mult.append(mult)

        # Vectorize data
        X_demo = d_vect.fit_transform(X_demo_lst)  # X_demo is now a numpy array
        X_prod = tfidf_vect.fit_transform(X_prod_lst)  # X_prod is now a CSR sparse matrix

        # Add batch result to output matrix
        if X is not None:
            X_t = hstack([csr_matrix(X_demo), X_prod], format="csr")
            try:
                X = vstack([X, X_t], format="csr")
            except ValueError:
                break
        else:
            # Initialize X
            X = hstack([csr_matrix(X_demo), X_prod], format="csr")

        batch_num += 1

    for como, mult in zip(PROD_COMO, demo_mult):
        val = como * mult
        if val < 6:
            y.append(0)
        elif val < 12:
            y.append(1)
        else:
            y.append(2)

    print('Storing vectorized data and training labels')
    # Flatten CSR sparse matrix to strings
    model = {
        'X': X,
        'y': y,
        'd_vect': d_vect,
        'tfidf_vect': tfidf_vect,
        'vocabulary': vocabulary
    }

    print("Saving model data to disk for next time")
    # Insert the model into the model database
    MODEL_DB.create_file(pdumps(model, protocol=2), filename="ml_people_data")
    # Save model data to disk
    with open(ppl_model_data, "wb") as pickle_out:
        pdump(model, pickle_out)
    print('[SUCCESS] People model data post-processed and stored')


def test_estimators(est_dicts, model):

    # Runs grid search on each estimator and records the best score
    # and standard deviation

    X = model['X']
    y = model['y']

    best_est_res = {}  # Stores estimator performance data
    for est_dict in est_dicts:
        print("Running grid search on {} estimator".format(est_dict['name']))
        grid = GridSearchCV(
            est_dict['callable'],
            est_dict['params'],
            n_jobs=-1,
            scoring='f1_micro',
            verbose=True)
        grid.fit(X, y)
        score, std_dev, est_call = (
            grid.best_score_,
            np.mean(grid.cv_results_['std_test_score']),
            grid.best_estimator_)
        best_est_res[est_dict['name']] = [score, std_dev, est_call]
    return best_est_res


def plot_best_estimator(estimator_results, custom_axis=None):
    # Plot the best score of each estimator
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111)
    xs = range(len(estimator_results))
    color = iter(cm.rainbow(np.linspace(0, 1, len(xs))))
    labels = list(estimator_results.keys())

    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(labels)

    for i, (clf, result) in enumerate(estimator_results.items()):
        c = next(color)
        ax.errorbar(
            i,
            1.0 * result[0],
            yerr=result[1],
            fmt='--o',
            color=c,
            capsize=5,
            label=labels[i]
            )
        ax.annotate(
            "{:.2f}\n(+/-{:.2E})".format(1.0 * result[0], result[1]),
            (i+0.05, 1.0 * result[0]))

    plt.title("Estimator F1 Score with Average Standard Deviation")
    plt.legend()
    for tick in ax.get_xticklabels():
        tick.set_rotation(50)
    if custom_axis is not None:
        plt.axis(custom_axis)
    plt.savefig("mm_test_result.png")


def optimize_product_model(host, port):
    prod_model_data = 'prod_model_data.pickle'
    result_data = 'estimator_results.pickle'
    print("Loading model data")
    try:
        with open(prod_model_data, "rb") as pickle_in:
            model = pload(pickle_in)
        print('Loaded from Pickle')
    except Exception as e:
        print("Loading from database...", e)
        x = MODEL_DB.read_file('ml_product_data').sort("uploadDate", -1).limit(1)
        model = pload(x[0])
        print("Saving model data to disk for next time")
        with open(prod_model_data, "wb") as pickle_out:
            pdump(model, pickle_out)

    print("Running gridsearchCV")

    estimator_results = test_estimators(est_dicts, model)
    plot_best_estimator(estimator_results)
    print(
        "Saving gridsearchCV results, explore by un-pickling",
        result_data,
        "with an IPython shell or python program.")
    with open(result_data, "wb") as pickle_out:
        pdump(estimator_results, pickle_out)


def optimize_people_model(host, port):
    ppl_model_data = 'ppl_model_data.pickle'
    result_data = 'estimator_results.pickle'
    print("Loading model data")
    try:
        with open(ppl_model_data, "rb") as pickle_in:
            model = pload(pickle_in)
        print('Loaded from Pickle')
    except Exception as e:
        print("Loading from database...", e)
        x = MODEL_DB.read_file('ml_product_data').sort("uploadDate", -1).limit(1)
        model = pload(x[0])
        print("Saving model data to disk for next time")
        with open(ppl_model_data, "wb") as pickle_out:
            pdump(model, pickle_out)

    print("Running gridsearchCV")

    estimator_results = test_estimators(est_dicts, model)
    plot_best_estimator(estimator_results)
    print(
        "Saving gridsearchCV results, explore by un-pickling",
        result_data,
        "with an IPython shell or python program.")
    with open(result_data, "wb") as pickle_out:
        pdump(estimator_results, pickle_out)


def initialize_connections(host, port):
    # Connect to database
    global PEOPLE_DB
    global PRODUCTS_DB
    global INGREDIENTS_DB
    global TEST_DB
    global COMODEGENIC_DB
    global MODEL_DB

    PEOPLE_DB = DB_CRUD(host, port, db='capstone', col='people')
    PRODUCTS_DB = DB_CRUD(host, port, db='capstone', col='products')
    INGREDIENTS_DB = DB_CRUD(host, port, db='capstone', col='ingredients')
    TEST_DB = DB_CRUD(host, port, db='capstone', col='testing')
    COMODEGENIC_DB = DB_CRUD(host, port, db='capstone', col='comodegenic')
    MODEL_DB = DB_CRUD(host, port, db='capstone', col='model')


def build_opt_model(host='localhost', port=27017, bld_model='', model_opt=''):
    # initializing the MongoClient, this helps to
    # access the MongoDB databases and collections

    # Connect to databases
    initialize_connections(host, port)

    if bld_model == 'prod':
        build_product_model(host, port)
    elif bld_model == 'ppl':
        build_people_model(host, port)
    elif bld_model:
        print('No action performed, bld_model must one of "prod" or "ppl"')

    if model_opt == 'prod':
        optimize_product_model(host, port)
    elif model_opt == 'ppl':
        optimize_people_model(host, port)
    elif model_opt:
        print('No action performed, model_opt must one of "prod" or "ppl"')
