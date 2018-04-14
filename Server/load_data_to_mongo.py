#!/usr/bin/env python3
'''
Loads data into MongoDB database and tests DB functionality
Author: Sombiri Enwemeka
'''

import argparse
from db_crud import DB_CRUD
from db_object import DB_Object, JSONEncoder
from test_db import database_test, query_yes_no
from build_db import build_db, generate_people
from model_ops import build_opt_model
import json

HOST_NAME = 'localhost'
PORT_NUMBER = 27017

INGREDIENT_FILE = 'ewg_ingredients.json'
PRODUCT_FILE = 'ewg_products.json'
CMDGNC_FILE = 'comodegenic.json'

PEOPLE_DB = None
PRODUCTS_DB = None
INGREDIENTS_DB = None
TEST_DB = None
COMODEGENIC_DB = None
MODEL_DB = None


def destroy_everything(host, port):
    print("Erasing all data")
    print("Erasing people database")
    PEOPLE_DB.nuke()
    print("Erasing products database")
    PRODUCTS_DB.nuke()
    print("Erasing ingredients database")
    INGREDIENTS_DB.nuke()
    print("Erasing testing database")
    TEST_DB.nuke()
    print("Erasing comodegenic database")
    COMODEGENIC_DB.nuke()
    print("Erasing model database")
    MODEL_DB.nuke()
    print("All databases removed")


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

    print("People database stats:")
    print_stat(PEOPLE_DB.stats())
    print("Products database stats:")
    print_stat(PRODUCTS_DB.stats())
    print("Ingredients database stats:")
    print_stat(INGREDIENTS_DB.stats())
    print("Testing database stats:")
    print_stat(TEST_DB.stats())
    print("Comodegenic database stats:")
    print_stat(COMODEGENIC_DB.stats())


def dump_db_to_json(host, port, dump_db):
    valid = ["people", "products", "ingredients", "testing", "comodegenic", "all"]
    repos = [PEOPLE_DB, PRODUCTS_DB, INGREDIENTS_DB, TEST_DB, COMODEGENIC_DB]
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


def main(**kwargs):
    # Get arguments
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

    # Connect to databases
    initialize_connections(host, port)

    if test:
        test_db = database_test(host, port, db='capstone', col='testing')
        test_db.test_db()

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

    if bld_model or model_opt:
        build_opt_model(host, port, bld_model, model_opt)

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
