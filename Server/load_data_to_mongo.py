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
from db_object import DB_Object

HOST_NAME = 'localhost'
PORT_NUMBER = 27017

INGREDIENT_FILE = 'ewg_ingredients.json'
PRODUCT_FILE = 'ewg_products.json'


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
        tmp_project = DB_Object.build_from_json(p)
        print("ID = {} | Title = {} | Price = {}".format(
            tmp_project._id,
            tmp_project.title,
            tmp_project.price))
    if not at_least_one_item:
        print("No items in the database")


def test_create(repository, new_object):
    print("\n\nSaving new_object to database")
    repository.create(new_object)
    print("new_object saved to database")
    print("Loading new_object from database")
    db_objects = repository.read(_id=new_object._id)
    for p in db_objects:
        project_from_db = DB_Object.build_from_json(p)
        print("new_object = {}".format(project_from_db.get_as_json()))


def test_update(repository, new_object):
    print("\n\nUpdating new_object in database")
    repository.update(new_object)
    print("new_object updated in database")
    print("Reloading new_object from database")
    db_objects = repository.read(_id=new_object._id)
    for p in db_objects:
        project_from_db = DB_Object.build_from_json(p)
        print("new_object = {}".format(project_from_db.get_as_json()))


def test_delete(repository, new_object):
    print("\n\nDeleting new_object from database")
    repository.delete(new_object)
    print("new_object deleted from database")
    print("Trying to reload new_object from database")
    db_objects = repository.read(_id=new_object._id)
    found = False
    for p in db_objects:
        found = True
        project_from_db = DB_Object.build_from_json(p)
        print("new_object = {}".format(project_from_db.get_as_json()))

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
        tmp_project = DB_Object.build_from_json(p)
        print("ID = {} | Title = {} | Price = {}".format(
            tmp_project._id,
            tmp_project.title,
            tmp_project.price))
    if not at_least_one_item:
        print("[SUCCESS] No items in the database")
    else:
        print("[FAILED] test_delete_all")


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
    new_object = DB_Object.build_from_json(json_data)
    test_create(test_db, new_object)

    #update new_object and read back from database
    new_object.price = 350
    test_update(test_db, new_object)

    #delete new_object and try to read back from database
    test_delete(test_db, new_object)

    #Test nuking and reading anything back from database
    for i in range(3):
        test_db.create(DB_Object.build_from_json(json_data))
    test_delete_all(test_db)


def build_db(host, port, **kwargs):
    import json
    # Connect to the reequired databases
    products_db = DB_CRUD(host, port, db='capstone', col='products')
    ingredients_db = DB_CRUD(host, port, db='capstone', col='ingredients')
    i_path = kwargs.get('i_path', '')
    p_path = kwargs.get('p_path', '')

    # Open files and load JSON data, exit if unsuccesful
    print("Attempting to open .json files.")
    try:
        i_f = open(i_path, 'rb')
        p_f = open(p_path, 'rb')
    except IOError as e:
        print(e)
        exit()
    with i_f:
        ingredients_dict = json.load(i_f)
        ing_ins_len = len(ingredients_dict)
    with p_f:
        products_dict = json.load(p_f)
        prod_ins_len = len(products_dict)

    # Clean and load ingredients into ingredient database
    print("Populating ingredients")
    for ingredient_id in list(ingredients_dict.keys()):
        # Remove the old id entry from ingredients_dict
        # This is to avoid storing redundant info in the DB, ingredient entries will still
        # be accessible using the ingredient_id when the product entries are added
        del(ingredients_dict[ingredient_id]['ingredient_id'])
        # Create DB object from ingredient
        new_ingredient = DB_Object.build_from_json(ingredients_dict[ingredient_id])
        # Add the new mongoDB id to the existing ingredients dictionary
        ingredients_dict[ingredient_id]['_id'] = new_ingredient['_id']
        # Insert the ingredient into the database
        ingredients_db.create(new_ingredient)

    print("Populating products")
    for product_id in list(products_dict.keys()):
        # Convert ingredient list IDs to Mongo DB object IDs
        new_ing_ids = []
        for ingredient_id in products_dict[product_id].get('ingredient_list', []):
            new_ing_id = ingredients_dict.get(ingredient_id, {}).get('_id', None)
            if new_ing_id:
                new_ing_ids.append(new_ing_id)
            else:
                raise KeyError(
                    "Check scraper, key should exist in ingredients JSON!\nKey: '{}'".format(
                        ingredient_id))
        if new_ing_ids:
            products_dict[product_id]['ingredient_list'] = new_ing_ids
        # Create DB object from product
        new_product = DB_Object.build_from_json(products_dict[product_id])
        # Insert the product into the database
        products_db.create(new_product)

    # Test the build
    print("Testing data integrity")
    ing_read_len = ingredients_db.read().count()
    prod_read_len = products_db.read().count()

    print("Ingredients inserted: {}  Ingredients read: {}".format(ing_ins_len, ing_read_len))
    print("Products inserted: {}  Products read: {}".format(prod_ins_len, prod_read_len))

    if ing_read_len != ing_ins_len or prod_read_len != prod_ins_len:
        # Nuke databases to prevent mismatch on retry
        ingredients_db.nuke()
        products_db.nuke()
        raise Exception("[FAIL] The number of inserted items does not match!")
    print("[SUCCESS] Database is populated")


def generate_people(host, port):
    import random
    import names
    races = [
        'American Indian',
        'Asian',
        'Black',
        'Pacific Islander',
        'White']

    people_db = DB_CRUD(host, port, db='capstone', col='people')
    products_db = DB_CRUD(host, port, db='capstone', col='products')
    ingredients_db = DB_CRUD(host, port, db='capstone', col='ingredients')


def destroy_everything(host, port):
    print("Erasing all data")
    people_db = DB_CRUD(host, port, db='capstone', col='people')
    products_db = DB_CRUD(host, port, db='capstone', col='products')
    ingredients_db = DB_CRUD(host, port, db='capstone', col='ingredients')
    test_db = DB_CRUD(host, port, db='capstone', col='testing')
    print("Erasing people database")
    ppl_res = people_db.nuke()
    print("Erased {} entries", ppl_res.deleted_count)
    print("Erasing products database")
    prod_res = products_db.nuke()
    print("Erased {} entries", prod_res.deleted_count)
    print("Erasing ingredients database")
    ing_res = ingredients_db.nuke()
    print("Erased {} entries", ing_res.deleted_count)
    print("Erasing testing database")
    test_res = test_db.nuke()
    print("Erased {} entries", test_res.deleted_count)
    print("Erasure complete\nResults:")


def main(**kwargs):
    host = kwargs.get('host', None)
    port = kwargs.get('port', None)
    test = kwargs.get('test', False)
    generate = kwargs.get('generate', False)
    build = kwargs.get('build', False)
    i_path = kwargs.get('ingredients', None)
    p_path = kwargs.get('products', None)
    nuke_all = kwargs.get('nuke', False)

    if test:
        test_db(host, port)

    if generate:
        generate_people(host, port)

    if build:
        build_db(host, port, i_path=i_path, p_path=p_path)

    if nuke_all:
        nuke_qstn = '[WARNING] This will erase everything in the databases. Continue?'
        if query_yes_no(nuke_qstn, default='no'):
            destroy_everything(host, port)
        else:
            print("No action taken")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--host', help='Server hostname', default=HOST_NAME)
    parser.add_argument('-p', '--port', help='Server port', default=PORT_NUMBER)
    parser.add_argument('-n', '--nuke', help='Erase all database data', action='store_true')
    parser.add_argument(
        '--ingredients',
        help='Specify ingredients JSON file',
        default=INGREDIENT_FILE)
    parser.add_argument(
        '--products',
        help='Specify products JSON file',
        default=PRODUCT_FILE)
    parser.add_argument('-t', '--test', help='Run DB connection tests', action='store_true')
    parser.add_argument(
        '-g',
        '--generate',
        help='Fill the people DB with automatically generated data',
        action='store_true')
    parser.add_argument(
        '-b',
        '--build',
        help='Populate the database with data from JSON files',
        action='store_true')
    args = parser.parse_args()

    main(**vars(args))
