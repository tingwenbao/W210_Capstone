#!/usr/bin/env python3
'''
Loads data into MongoDB database and tests DB functionality
Author: Sombiri Enwemeka

References code from Greg Bogdan
source: https://www.freelancer.com/community/articles/crud-operations-in-mongodb-using-python
'''

import sys
import argparse
from db_crud import DB_CRUD
from db_object import DB_Object, JSONEncoder
import json
import numpy as np
from bson.binary import Binary
from pymongo import TEXT, ASCENDING, DESCENDING
import base64
from pickle import dumps as pdumps
from pickle import dump as pdump
from pickle import load as pload
from faker import Faker
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    make_scorer)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import (
    HuberRegressor,
    PassiveAggressiveRegressor,
    TheilSenRegressor,
    RANSACRegressor)
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    IsolationForest)
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.pyplot import cm
from ml_test_params import *

# Fake info generator
fake = Faker()

HOST_NAME = 'localhost'
PORT_NUMBER = 27017

INGREDIENT_FILE = 'ewg_ingredients.json'
PRODUCT_FILE = 'ewg_products.json'
CMDGNC_FILE = 'comodegenic.json'

NAMES_SEEN = set()


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def load_all_items_from_database(repository):
    print("Loading all items from database:")
    db_objects = repository.read()
    at_least_one_item = False
    for p in db_objects:
        at_least_one_item = True
        tmp_project = DB_Object.build_from_dict(p)
        print("ID = {} | Title = {} | Price = {}".format(
            tmp_project._id,
            tmp_project.title,
            tmp_project.price))
    if not at_least_one_item:
        print("No items in the database")


def test_create(repository, new_object):
    print("\n\nSaving new_object to database")
    result = repository.create(new_object)
    if result.acknowledged:
        new_object['_id'] = result.inserted_id
    else:
        print("[FAILED] Could not save object")
    print("new_object saved to database")
    print("Loading new_object from database")
    db_objects = repository.read({'_id': new_object._id})
    for p in db_objects:
        project_from_db = DB_Object.build_from_dict(p)
        print("new_object = {}".format(project_from_db.get_as_dict()))


def test_update(repository, new_object):
    print("\n\nUpdating new_object in database")
    repository.update(new_object)
    print("new_object updated in database")
    print("Reloading new_object from database")
    db_objects = repository.read({'_id': new_object._id})
    for p in db_objects:
        project_from_db = DB_Object.build_from_dict(p)
        print("new_object = {}".format(project_from_db.get_as_dict()))


def test_delete(repository, new_object):
    print("\n\nDeleting new_object from database")
    repository.delete(new_object)
    print("new_object deleted from database")
    print("Trying to reload new_object from database")
    db_objects = repository.read({'_id': new_object._id})
    found = False
    for p in db_objects:
        found = True
        project_from_db = DB_Object.build_from_dict(p)
        print("new_object = {}".format(project_from_db.get_as_dict()))

    if not found:
        print("Item with id = {} was not found in the database".format(new_object._id))


def test_delete_all(repository):
    print("\n\nDeleting EVERYTHING from database")
    repository.nuke()
    print("NUKED database")
    print("Trying to reload new_object from database")
    db_objects = repository.read()
    at_least_one_item = False
    for p in db_objects:
        at_least_one_item = True
    if at_least_one_item:
        print("[FAILED] Items still in " + repository.collection + " database")
    else:
        print("[SUCCESS] No items in " + repository.collection + " database")


def test_db(host, port):
    '''
        Test database CRUD ops
    '''
    test_db = DB_CRUD(host, port, db='capstone', col='testing')

    #display all items from DB
    load_all_items_from_database(test_db)

    #create new_object and read back from database
    json_data = {
        "title": "Wordpress website for Freelancers",
        "description": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc molestie. ",
        "price": 250,
        "assigned_to": "John Doe"}
    new_object = DB_Object.build_from_dict(json_data)
    test_create(test_db, new_object)

    #update new_object and read back from database
    new_object.price = 350
    test_update(test_db, new_object)

    #delete new_object and try to read back from database
    test_delete(test_db, new_object)

    #Test nuking and reading anything back from database
    for i in range(3):
        test_db.create(DB_Object.build_from_dict(json_data))
    test_delete_all(test_db)


def build_db(host, port, **kwargs):
    # Connect to the reequired databases
    products_db = DB_CRUD(host, port, db='capstone', col='products')
    ingredients_db = DB_CRUD(host, port, db='capstone', col='ingredients')
    comodegenic_db = DB_CRUD(host, port, db='capstone', col='comodegenic')
    i_path = kwargs.get('i_path', '')
    p_path = kwargs.get('p_path', '')
    c_path = kwargs.get('c_path', '')
    score_max = kwargs.get('score_max', False)

    # Make sure user wants to destroy existing DB
    db_qstn = (
        '[WARNING] This will erase the products, ingredients, '
        'and comodegenic items databases. Continue?')
    if not query_yes_no(db_qstn, default='no'):
        print("No actions taken")
        return

    # Drop databases
    print("Deleting products database")
    products_db.nuke()
    print("Deleting ingredients database")
    ingredients_db.nuke()
    print("Deleting comodegenic database")
    comodegenic_db.nuke()

    # Open files and load JSON data, exit if unsuccesful
    print("Attempting to open .json files.")
    try:
        i_f = open(i_path, 'rb')
        p_f = open(p_path, 'rb')
        c_f = open(c_path, 'rb')
    except IOError as e:
        print(e)
        exit()
    with i_f:
        ingredients_dict = json.load(i_f)
        ing_ins_len = len(ingredients_dict)
    with p_f:
        products_dict = json.load(p_f)
        prod_ins_len = len(products_dict)
    with c_f:
        cmdgnc_list = json.load(c_f)
        print("Populating comodegenic information")
        #cmdgnc_dict = {entry['ingredient']: entry for entry in cmdgnc_list}
        for entry in cmdgnc_list:
            # Create DB object from product
            new_entry = DB_Object.build_from_dict(entry)
            # Insert the product into the database
            comodegenic_db.create(new_entry)
        comodegenic_db.createIndex([('ingredient', TEXT)])

    # Clean and load ingredients into ingredient database
    print("Populating ingredients")
    for ingredient_id in list(ingredients_dict.keys()):
        ingredient = ingredients_dict[ingredient_id]
        # Remove the old id entry from ingredients_dict
        # This is to avoid storing redundant info in the DB, ingredient entries will still
        # be accessible using the ingredient_id when the product entries are added
        del(ingredient['ingredient_id'])
        # Get comodegenic info
        search_term = '"' + ingredient.get('ingredient_name', '') + '"'
        db_objects = comodegenic_db.read(
            {'$text': {"$search": search_term}})
        entries = [DB_Object.build_from_dict(entry) for entry in db_objects]

        # Try to find ingredient in comodegenic DB, fall back to synonyms if necessary
        if entries:
            ingredient['comodegenic'] = int(entries[0]['level'])
        else:
            for synonym in ingredient.get('synonym_list', []):
                search_term = '"' + synonym + '"'
                db_objects = comodegenic_db.read(
                    {'$text': {"$search": search_term}})
                entries = [DB_Object.build_from_dict(entry) for entry in db_objects]
                if entries:
                    ingredient['comodegenic'] = int(entries[0]['level'])
                    break
        # Set null value for ingredients without comodegenic score information
        if not 'comodegenic' in ingredient:
            ingredient['comodegenic'] = None

        # Normalize text fields
        ingredient['ingredient_name'] = ingredient.get('ingredient_name', '').strip().lower()
        norm_synonyms = []
        synonym_list = ingredient.get('synonym_list', [])
        for synonym in synonym_list:
            norm_synonyms.append(synonym.strip().lower())
        if synonym_list:
            ingredient['synonym_list'] = synonym_list

        # Create DB object from ingredient
        new_ingredient = DB_Object.build_from_dict(ingredient)

        # Insert the ingredient into the database
        db_op_res = ingredients_db.create(new_ingredient)

        # Add the new mongoDB id to the existing ingredients dictionary
        # if the insertion was successful
        if db_op_res.acknowledged:
            ingredient['_id'] = db_op_res.inserted_id
        else:
            err_msg = (
                "[FAIL] Database insertion for "
                + str(new_ingredient)
                + " was unsuccessful")
            raise Exception(err_msg)

    print("Populating products")
    for product_id in list(products_dict.keys()):
        # Convert ingredient list IDs to Mongo DB object IDs
        new_ing_ids = []
        product = products_dict[product_id]
        for ingredient_id in product.get('ingredient_list', []):
            new_ing_id = ingredients_dict.get(ingredient_id, {}).get('_id', None)
            if new_ing_id:
                new_ing_ids.append(new_ing_id)
                # Set product comodegenic score
                # Determine whether comodegenic scores are calculated using
                # ingredient max comodegenic score or sum of ingredient comodegenic scores
                ing_como = ingredients_dict[ingredient_id].get('comodegenic', 0)
                prod_como = product.get('comodegenic', 0)

                if score_max:
                    product['comodegenic'] = max(prod_como, ing_como)
                else:
                    product['comodegenic'] = prod_como + ing_como if ing_como else prod_como

            else:
                raise KeyError(
                    "Check scraper, key should exist in ingredients JSON!\nKey: '{}'".format(
                        ingredient_id))
        if new_ing_ids:
            product['ingredient_list'] = new_ing_ids
        # Set null value for products without comodegenic score information
        if not 'comodegenic' in product:
            product['comodegenic'] = None
        # Remove old style product id
        del(product['product_id'])
        # Create DB object from product
        new_product = DB_Object.build_from_dict(product)
        # Insert the product into the database
        products_db.create(new_product)

    # Test the build
    print("Testing data integrity")
    ing_read_len = ingredients_db.read().count()
    prod_read_len = products_db.read().count()

    print("Ingredients inserted: {}  Ingredients read: {}".format(ing_ins_len, ing_read_len))
    print("Products inserted: {}  Products read: {}".format(prod_ins_len, prod_read_len))

    if ing_read_len != ing_ins_len or prod_read_len != prod_ins_len:
        raise Exception("[FAIL] The number of inserted items does not match!")

    print("Creating search indexes")
    ingredients_db.createIndex(
        [('ingredient_name', TEXT), ('synonym_list', TEXT)],
        weights={'ingredient_name': 4},
        default_language='english')
    products_db.createIndex(
        [('product_name', TEXT)],
        default_language='english')
    products_db.createIndex(
        [('comodegenic', DESCENDING)],
        default_language='english')

    print("[SUCCESS] Database is populated")


def generate_age_acne_lists(num_ages):
    ret_age = []
    ret_acne = []
    ages = [age for age in range(0, 90, 5)]
    acne = [True, False]
    age_bin_probs = [
        0.065, 0.066, 0.067, 0.071, 0.070, 0.068, 0.065, 0.065, 0.068,
        0.074, 0.072, 0.064, 0.054, 0.040, 0.030, 0.024, 0.019, 0.018]
    acne_bin_probs = [
        0.025, 0.05, 0.6, 0.9, 0.8, 0.6, 0.5, 0.3, 0.2, 0.18,
        0.16, 0.15, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02]
    for i in range(num_ages):
        age = np.random.choice(ages, p=age_bin_probs) + np.random.choice(5)
        acne_prob = acne_bin_probs[int(len(acne_bin_probs) * age / 90)]
        ret_age.append(int(age))
        ret_acne.append(bool(np.random.choice(acne, p=[acne_prob, 1-acne_prob])))
    return ret_age, ret_acne


def get_unique_username(full_name):
    '''
    Create first initial + last name username,
    ensure it is unique by adding numbers to it. Expand
    to letter if that fails.
    '''
    name_list = full_name.split(' ')
    try:
        uname = name_list[0][0] + name_list[1]
    except IndexError:
        uname = name_list[0][0] + str(np.random.choice(1000000))

    apnd = ''
    for i in range(1000):
        if uname not in NAMES_SEEN:
            # Name is unique, we're done
            NAMES_SEEN.add(uname)
            return uname.lower()
        else:
            # Name is not unique, randomize it a bit
            uname = uname.strip(apnd)
            apnd = str(np.random.choice(1000000))
            uname = uname + apnd
    apnd = fake.password(length=5, special_chars=False, upper_case=False)
    uname = uname + apnd
    NAMES_SEEN.add(uname)
    return uname.lower()


def get_sex_name(s):
    if s == 'male':
        return fake.first_name_male() + ' ' + fake.last_name_male()
    else:
        return fake.first_name_female() + ' ' + fake.last_name_female()


def generate_people(host, port, num_generate_people=10000):

    # Variables
    races = [
        'American Indian',
        'Asian',
        'Black',
        'Pacific Islander',
        'White',
        'mixed_other']
    birth_sexes = [
        'female',
        'male']
    skin_types = [
        'normal',
        'oily',
        'dry']

    # Probabilities
    race_probs = [0.009, 0.048, 0.126, 0.002, 0.724, 0.091]
    sex_probs = [0.508, 0.492]
    skin_probs = [1.0/3, 1.0/3, 1.0/3]

    # Make sure user wants to destroy existing DB
    ppl_qstn = '[WARNING] This will erase the people database. Continue?'
    if not query_yes_no(ppl_qstn, default='no'):
        print("No actions taken")
        return

    # Get number of people to generate
    try:
        usr_input = int(input("# people to generate: "))
        num_generate_people = usr_input
    except ValueError:
        print(
            "Invalid input, using default value",
            num_generate_people)
        pass

    # Connect to DBs
    people_db = DB_CRUD(host, port, db='capstone', col='people')
    products_db = DB_CRUD(host, port, db='capstone', col='products')

    print("Nuking people database")
    people_db.nuke()

    print("Creating search indexes")
    people_db.createIndex(
        [('user_name', ASCENDING)],
        unique=True,
        default_language='english')

    # Generate random people data
    print("Generating race data")
    ppl_race = np.random.choice(races, num_generate_people, p=race_probs)
    print("Generating sex data")
    ppl_sex = np.random.choice(birth_sexes, num_generate_people, p=sex_probs)
    print("Generating age and acne data")
    ppl_ages, ppl_acne = generate_age_acne_lists(num_generate_people)
    print("Generating skin data")
    ppl_skins = np.random.choice(skin_types, num_generate_people, p=skin_probs)
    print("Generating names")
    ppl_names = [get_sex_name(s) for s in ppl_sex]
    print("Generating usernames")
    ppl_unames = [get_unique_username(full_name) for full_name in ppl_names]
    print("Generating user authentications")
    ppl_auths = [base64.b64encode(str(u_name+":1234").encode()).decode() for u_name in ppl_unames]

    # Generate dict of people
    print("Creating list of people dicts")
    fields = ['name', 'race', 'birth_sex', 'age', 'acne', 'skin', 'auth', 'user_name']
    p_data = zip(ppl_names, ppl_race, ppl_sex, ppl_ages, ppl_acne, ppl_skins, ppl_auths, ppl_unames)
    p_list = [dict(zip(fields, d)) for d in p_data]

    # Get comodegenic products
    print("Getting list of comodegenic products")
    db_objects = products_db.read({'comodegenic': {"$gt": 0}})
    products = [DB_Object.build_from_dict(p) for p in db_objects]

    # Set scaling for comodogenic-ness of products
    # The scale value is 1 divided by the maximum comodegenic score
    # in the products database which works regardless of the scoring
    # method used when building the db.
    prod_filt = {'comodegenic': {'$type': 'int'}}
    prod_prjctn = {'comodegenic': True}
    db_objects = products_db.read(
        prod_filt,
        projection=prod_prjctn,
        sort=[("comodegenic", DESCENDING)],
        limit=1)
    como_scale = 1.0 / DB_Object.build_from_dict(db_objects[0])['comodegenic']

    print("Adding people to database")
    # Populate acne causing products for each person
    for person in p_list:
        if person['acne']:
            p_products = []
            for i in range(np.random.choice(5)):  # Randomly choose 0 to 5 products
                rand_idx = np.abs(np.random.choice(len(products))-1)
                prod_como = products[rand_idx]['comodegenic']
                probs = [como_scale * prod_como, 1 - (como_scale * prod_como)]
                # Add product to person based on comodegenic score
                if np.random.choice([True, False], p=probs):
                    p_products.append(products[rand_idx]['_id'])
            person['acne_products'] = p_products
        else:
            person['acne_products'] = []
        # Add person to data base
        new_person = DB_Object.build_from_dict(person)
        people_db.create(new_person)

    print("[SUCCESS] people database is populated")


def destroy_everything(host, port):
    print("Erasing all data")
    people_db = DB_CRUD(host, port, db='capstone', col='people')
    products_db = DB_CRUD(host, port, db='capstone', col='products')
    ingredients_db = DB_CRUD(host, port, db='capstone', col='ingredients')
    test_db = DB_CRUD(host, port, db='capstone', col='testing')
    comodegenic_db = DB_CRUD(host, port, db='capstone', col='comodegenic')

    print("Erasing people database")
    ppl_res = people_db.nuke()
    print("Erased:", ppl_res.deleted_count)
    print("Erasing products database")
    prod_res = products_db.nuke()
    print("Erased:", prod_res.deleted_count)
    print("Erasing ingredients database")
    ing_res = ingredients_db.nuke()
    print("Erased:", ing_res.deleted_count)
    print("Erasing testing database")
    test_res = test_db.nuke()
    print("Erased:", test_res.deleted_count)
    print("Erasing comodegenic database")
    comodegenic_res = comodegenic_db.nuke()
    print("Erased:", comodegenic_res.deleted_count)
    print("Erasure complete\nResults:")


def print_stat(col_stats):
    '''
    prints db collection statistics
    '''
    if type(col_stats) is not dict:
        return
    for (k, v) in col_stats.items():
        if k == 'wiredTiger' or k == 'indexSizes' or k == 'indexDetails':
            continue
        elif k == 'info':
            print("\t" + v)
        else:
            print("\t" + k + ":", v)


def display_db_stats(host, port):
    num_db = 5
    print("Database stats:")
    print("There are", num_db, "collections.")

    people_db = DB_CRUD(host, port, db='capstone', col='people')
    products_db = DB_CRUD(host, port, db='capstone', col='products')
    ingredients_db = DB_CRUD(host, port, db='capstone', col='ingredients')
    test_db = DB_CRUD(host, port, db='capstone', col='testing')
    comodegenic_db = DB_CRUD(host, port, db='capstone', col='comodegenic')

    print("People database stats:")
    print_stat(people_db.stats())
    print("Products database stats:")
    print_stat(products_db.stats())
    print("Ingredients database stats:")
    print_stat(ingredients_db.stats())
    print("Testing database stats:")
    print_stat(test_db.stats())
    print("Comodegenic database stats:")
    print_stat(comodegenic_db.stats())


def dump_db_to_json(host, port, dump_db):
    valid = ["people", "products", "ingredients", "testing", "comodegenic", "all"]

    # Connect to dbs
    people_db = DB_CRUD(host, port, db='capstone', col='people')
    products_db = DB_CRUD(host, port, db='capstone', col='products')
    ingredients_db = DB_CRUD(host, port, db='capstone', col='ingredients')
    test_db = DB_CRUD(host, port, db='capstone', col='testing')
    comodegenic_db = DB_CRUD(host, port, db='capstone', col='comodegenic')

    repos = [people_db, products_db, ingredients_db, test_db, comodegenic_db]
    out_list = {}

    # Input validation
    if dump_db is None or dump_db is "":
        return
    if dump_db not in valid:
        return

    # Dump the specified DB
    print("Dumping database/s: '" + dump_db + "'")
    db_objects = repos[0].read()
    at_least_one_item = False
    if dump_db == 'all':
        for repo in repos:
            db_objects = repo.read()
            at_least_one_item = False
            out_list[repo.collection] = []
            for p in db_objects:
                at_least_one_item = True
                out_list[repo.collection].append(DB_Object.build_from_dict(p))
            if not at_least_one_item:
                print("No items in ", repo.collection, " database")
    else:
        repo_idx = valid.index(dump_db)
        db_objects = repos[repo_idx].read()
        at_least_one_item = False
        out_list[repos[repo_idx].collection] = []
        for p in db_objects:
            at_least_one_item = True
            out_list[repos[repo_idx].collection].append(
                DB_Object.build_from_dict(p).get_as_dict())
        if not at_least_one_item:
            print("No items in ", repos[repo_idx].collection, " database")

    with open('db_dump_%s.json' % dump_db, 'w') as f:
        json.dump(out_list, f, cls=JSONEncoder)


def build_product_model(host, port, **kwargs):
    prod_model_data = 'prod_model_data.pickle'
    products_db = DB_CRUD(host, port, db='capstone', col='products')
    ingredients_db = DB_CRUD(host, port, db='capstone', col='ingredients')
    model_db = DB_CRUD(host, port, db='capstone', col='model')

    print("Loading products from database:")
    prod_filt = {'comodegenic': {'$type': 'int'}}  # Only return entries with comodegenic score
    prod_prjctn = {
        'ingredient_list': True,
        'comodegenic': True}
    db_objects = products_db.read(prod_filt, projection=prod_prjctn)
    products = [DB_Object.build_from_dict(p) for p in db_objects]

    # The vectorizer will ignore the following words
    stop_words = [
        '',
        'WATER',
        'GLYCERIN',
        'TITANIUM DIOXIDE',
        'IRON OXIDES',
        'BEESWAX',
        'METHYLPARABEN',
        'PROPYLPARABEN',
        'PROPYLENE GLYCOL',
        'PANTHENOL',
        'MICA']

    # Tokenizer for product ingredient lists
    def get_ingredients_as_list(product):
        '''
        Queries the ingredients DB for a given product's ingredient list
        and returns the ingredient list as a list of ingredient strings
        Note: The DB query is performed once using all ingredient object
        IDs simultaneously.
        '''
        fltr = {'_id': {'$in': product.get('ingredient_list', [])}}
        ing_prjctn = {'_id': False, 'ingredient_name': True}
        db_objects = ingredients_db.read(fltr, projection=ing_prjctn)
        return [DB_Object.build_from_dict(i).get('ingredient_name', '') for i in db_objects]

    print('Vectorizing product ingredient lists')
    vectorizer = TfidfVectorizer(
        tokenizer=get_ingredients_as_list,
        lowercase=False,
        stop_words=stop_words)
    X = vectorizer.fit_transform(products)
    y = [p['comodegenic'] for p in products]

    print('Storing vectorized data and training labels')
    # Flatten CSR sparse matrix to strings
    model = {
        'X': X,
        'y': y
    }

    print("Saving model data to disk for next time")
    # Insert the model into the model database
    model_db.create_file(pdumps(model, protocol=2), filename="ml_product_data")
    # Save model data to disk
    with open(prod_model_data, "wb") as pickle_out:
        pdump(model, pickle_out)
    print('[SUCCESS] Product model data post-processed and stored')


def get_ingredient_vocabulary(host, port, **kwargs):
    # Returns the set of all unique ingredient names including synonyms
    ingredients_db = DB_CRUD(host, port, db='capstone', col='ingredients')

    # Build list of all ingredient names
    ing_fltr = {}  # Get all ingredients
    ing_prjctn = {
        '_id': False,
        'ingredient_name': True,
        'synonym_list': True}
    db_objects = ingredients_db.read(ing_fltr, projection=ing_prjctn)
    ingredients = [DB_Object.build_from_dict(i) for i in db_objects]
    ret = set()
    for ingredient in ingredients:
        ret.update([ingredient.get('ingredient_name', '')])
        for synonym in ingredient.get('synonym_list', []):
            ret.update([ingredient.get('ingredient_name', '')])
    return ret


def build_people_model(host, port, **kwargs):
    ppl_model_data = 'ppl_model_data.pickle'
    products_db = DB_CRUD(host, port, db='capstone', col='products')
    people_db = DB_CRUD(host, port, db='capstone', col='people')
    ingredients_db = DB_CRUD(host, port, db='capstone', col='ingredients')
    model_db = DB_CRUD(host, port, db='capstone', col='model')

    batch_size = kwargs.get('batch_size', 10000)
    vocabulary = get_ingredient_vocabulary(host, port)

    # The vectorizer will ignore the following words
    stop_words = [
        '',
        'WATER',
        'GLYCERIN',
        'TITANIUM DIOXIDE',
        'IRON OXIDES',
        'BEESWAX',
        'METHYLPARABEN',
        'PROPYLPARABEN',
        'PROPYLENE GLYCOL',
        'PANTHENOL',
        'MICA']

    # Tokenizer for ingredient lists
    def get_ingredients_as_list(p_list):
        '''
        Queries the products and ingredients DBs for ingredients contained within
        the products given by the input list of Object_Ids.
        Note: The each DB query is performed once using all object
        IDs simultaneously. This function performs no more than 2 queries when run.
        '''

        if not p_list:
            return []

        # Build list of ingredient ObjectIds contained in the p_list
        prod_fltr = {'_id': {'$in': p_list}}
        prod_prjctn = {'_id': False, 'ingredient_list': True}
        db_objects = products_db.read(prod_fltr, projection=prod_prjctn)
        # Get ObjectIds of all product ingredients
        ing_list = set()  # Using set eliminates duplicate values
        for i in db_objects:
            ing_list.update(DB_Object.build_from_dict(i).get('ingredient_list', ''))

        # Build list of all ingredient names
        ing_fltr = {'_id': {'$in': list(ing_list)}}
        ing_prjctn = {'_id': False, 'ingredient_name': True}
        db_objects = ingredients_db.read(ing_fltr, projection=ing_prjctn)
        return [DB_Object.build_from_dict(i).get('ingredient_name', '') for i in db_objects]

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
    db_objects = people_db.read(ppl_filt, projection=ppl_prjctn)

    y = []
    batch_num, pulled = (0, 0)
    X = None

    # Work in batches to build dataset
    while pulled <= db_objects.count(with_limit_and_skip=True):
        # Initialize
        X_demo_lst, X_prod_lst = ([], [])
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
        for i in range(len(people)):
            person = people[i]
            # Pull product ingredients info
            X_prod_lst.append(person.pop('acne_products'))

            # Pull acne indicator
            y.append(person.pop('acne'))

            # Pull demographic info
            X_demo_lst.append(person)

        # Vectorize data
        d_vect = DictVectorizer(sparse=False)
        X_demo = d_vect.fit_transform(X_demo_lst)  # X_demo is now a numpy array

        vectorizer = TfidfVectorizer(
            tokenizer=get_ingredients_as_list,
            lowercase=False,
            stop_words=stop_words,
            vocabulary=vocabulary)
        X_prod = vectorizer.fit_transform(X_prod_lst)  # X_prod is now a CSR sparse matrix

        # Add batch result to output matrix
        if X is not None:
            X_t = hstack([csr_matrix(X_demo), X_prod], format="csr")
            #print("X", X.shape, "X_t",X_t.shape,'X_demo',X_demo.shape,'X_prod',X_prod.shape)
            try:
                X = vstack([X, X_t], format="csr")
            except:
                import ipdb
                ipdb.set_trace()
        else:
            # Initialize X
            X = hstack([csr_matrix(X_demo), X_prod], format="csr")

        batch_num += 1

    print('Storing vectorized data and training labels')
    # Flatten CSR sparse matrix to strings
    model = {
        'X': X,
        'y': y
    }

    print("Saving model data to disk for next time")
    # Insert the model into the model database
    model_db.create_file(pdumps(model, protocol=2), filename="ml_people_data")
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
            scoring='accuracy',
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

    plt.title("Estimator Accuracy with Average Standard Deviation")
    plt.legend()
    for tick in ax.get_xticklabels():
        tick.set_rotation(50)
    if custom_axis is not None:
        plt.axis(custom_axis)
    plt.savefig("mm_test_result.png")


def optimize_product_model(host, port):
    model_db = DB_CRUD(host, port, db='capstone', col='model')
    prod_model_data = 'prod_model_data.pickle'
    result_data = 'estimator_results.pickle'
    print("Loading model data")
    try:
        with open(prod_model_data, "rb") as pickle_in:
            model = pload(pickle_in)
        print('Loaded from Pickle')
    except Exception as e:
        print("Loading from database...", e)
        x = model_db.read_file('ml_product_data').sort("uploadDate", -1).limit(1)
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
    model_db = DB_CRUD(host, port, db='capstone', col='model')
    ppl_model_data = 'ppl_model_data.pickle'
    result_data = 'estimator_results.pickle'
    print("Loading model data")
    try:
        with open(ppl_model_data, "rb") as pickle_in:
            model = pload(pickle_in)
        print('Loaded from Pickle')
    except Exception as e:
        print("Loading from database...", e)
        x = model_db.read_file('ml_product_data').sort("uploadDate", -1).limit(1)
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


def main(**kwargs):
    host = kwargs.get('host', None)
    port = kwargs.get('port', None)
    test = kwargs.get('test', False)
    generate = kwargs.get('generate', False)
    build = kwargs.get('build', False)
    bld_model = kwargs.get('build_model', '')
    model_opt = kwargs.get('model_opt', '')
    score_max = kwargs.get('score_max', False)
    i_path = kwargs.get('ingredients', None)
    p_path = kwargs.get('products', None)
    c_path = kwargs.get('como', None)
    nuke_all = kwargs.get('nuke', False)
    stats = kwargs.get('stats', False)
    dump_db = kwargs.get('dump', None)

    if test:
        test_db(host, port)

    if build:
        build_db(
            host,
            port,
            i_path=i_path,
            p_path=p_path,
            c_path=c_path,
            score_max=score_max)

    if generate:
        generate_people(host, port)

    if bld_model == 'prod':
        build_product_model(host, port)
    elif bld_model == 'ppl':
        build_people_model(host, port)

    if model_opt == 'prod':
        optimize_product_model(host, port)
    elif model_opt == 'ppl':
        optimize_people_model(host, port)

    if stats:
        display_db_stats(host, port)

    if nuke_all:
        nuke_qstn = '[WARNING] This will erase everything in the databases. Continue?'
        if query_yes_no(nuke_qstn, default='no'):
            destroy_everything(host, port)
        else:
            print("No action taken")

    if dump_db:
        dump_db_to_json(host, port, dump_db)

    # Inform user that they didn't select any options and provide help
    boolsum = (
        test
        + build
        + bool(bld_model)
        + generate
        + stats
        + nuke_all
        + bool(model_opt)
        + bool(dump_db))
    if not bool(boolsum):
        print(
            "The program didn't do anything because no options were selected, "
            "rerun with '-h' for help")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    select_one = parser.add_argument_group(
        'Required arguments (select at least one of the following)')
    parser.add_argument('-o', '--host', help='Server hostname', default=HOST_NAME)
    parser.add_argument('-p', '--port', help='Server port', default=PORT_NUMBER)
    select_one.add_argument('-n', '--nuke', help='Erase all database data', action='store_true')
    select_one.add_argument(
        '-s',
        '--stats',
        help='Display info about stored data',
        action='store_true')
    parser.add_argument(
        '--ingredients',
        help='Specify ingredients JSON file',
        default=INGREDIENT_FILE)
    parser.add_argument(
        '--products',
        help='Specify products JSON file',
        default=PRODUCT_FILE)
    parser.add_argument(
        '--como',
        help='Specify comodegenic JSON file',
        default=CMDGNC_FILE)
    select_one.add_argument('-t', '--test', help='Run DB connection tests', action='store_true')
    select_one.add_argument(
        '--score_max',
        help='Use max ingredient comodegenic score instead of summing all ingredient scores',
        action='store_true')
    select_one.add_argument(
        '-g',
        '--generate',
        help='Fill the people DB with automatically generated data',
        action='store_true')
    select_one.add_argument(
        '-b',
        '--build',
        help='Build the ingredients, products, and comodegenic databases.',
        action='store_true')
    select_one.add_argument(
        '--build_model',
        help=(
            'Build and store the ML model. Choose one of '
            '(ppl | prod)'))
    select_one.add_argument(
        '--model_opt',
        help=(
            'ML model optimization. Choose one of '
            '(ppl | prod)'))
    select_one.add_argument(
        '-d',
        '--dump',
        help=(
            'Dump specified DB to JSON, one of '
            '(all | people | testing | products | ingredients | comodegenic)'))
    args = parser.parse_args()

    main(**vars(args))
