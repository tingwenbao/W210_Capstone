#!/usr/bin/env python3
'''
Loads capstone project data into MongoDB database
Author: Sombiri Enwemeka
'''

from db_crud import DB_CRUD
from db_object import DB_Object
from test_db import query_yes_no
import json
import numpy as np
from pymongo import TEXT, ASCENDING, DESCENDING
import base64
from faker import Faker
from ml_test_params import *

# Fake info generator
fake = Faker()
NAMES_SEEN = set()


def build_db(host, port, **kwargs):
    # Get required file paths
    i_path = kwargs.get('i_path', '')
    p_path = kwargs.get('p_path', '')
    c_path = kwargs.get('c_path', '')
    score_max = kwargs.get('score_max', False)

    # Connect to the reequired databases
    products_db = DB_CRUD(host, port, db='capstone', col='products')
    ingredients_db = DB_CRUD(host, port, db='capstone', col='ingredients')
    comodegenic_db = DB_CRUD(host, port, db='capstone', col='comodegenic')

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
        weights={'ingredient_name': 10},
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

    # Connect to the required databases
    products_db = DB_CRUD(host, port, db='capstone', col='products')
    people_db = DB_CRUD(host, port, db='capstone', col='people')

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
    # 0 value comodegeinc scores are null data
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
        p_products = []
        for i in range(np.random.choice(10)):
            rand_idx = np.abs(np.random.choice(len(products))-1)
            prod_como = products[rand_idx]['comodegenic']
            probs = [como_scale * prod_como, 1 - (como_scale * prod_como)]
            if person['acne']:
                # If a person has acne, probabilisticly add 0 to 5 known
                # comodegenic products. Otherwise probabilisticly add
                # 0 to 5 non-comodegenic products
                if np.random.choice([True, False], p=probs):
                    p_products.append(products[rand_idx]['_id'])
            else:
                if np.random.choice([False, True], p=probs):
                    p_products.append(products[rand_idx]['_id'])
        person['acne_products'] = p_products
        #import ipdb
        #ipdb.set_trace()

        # Add person to data base
        new_person = DB_Object.build_from_dict(person)
        people_db.create(new_person)

    print("[SUCCESS] people database is populated")
