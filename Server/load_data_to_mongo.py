#!/usr/bin/env python3
'''
Loads data into mongoDB database

Credit to Greg Bogdan for initial code which was modified and generalized for this
project.
source: https://www.freelancer.com/community/articles/crud-operations-in-mongodb-using-python
'''

from pymongo import MongoClient
import json
import pandas as pd
import argparse
from db_crud import DB_CRUD
from db_object import DB_Object

HOST_NAME = 'localhost'
PORT_NUMBER = 27017


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
    print("\n\nDeleting All from database")
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


def main(**kwargs):
    host = kwargs.get('host', None)
    port = kwargs.get('port', None)
    test = kwargs.get('test', False)
    capstone_db = DB_CRUD(host, port, db='capstone', col='ingredients')

    if test:
        #display all items from DB
        load_all_items_from_database(capstone_db)

        #create new_object and read back from database
        json_data = {
            "title": "Wordpress website for Freelancers",
            "description": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc molestie. ",
            "price": 250,
            "assigned_to": "John Doe"}
        new_object = DB_Object.build_from_json(json_data)
        test_create(capstone_db, new_object)

        #update new_object and read back from database
        new_object.price = 350
        test_update(capstone_db, new_object)

        #delete new_object and try to read back from database
        test_delete(capstone_db, new_object)

        #Test nuking and reading anything back from database
        test_create(capstone_db, DB_Object.build_from_json(json_data))
        test_create(capstone_db, DB_Object.build_from_json(json_data))
        test_create(capstone_db, DB_Object.build_from_json(json_data))
        test_create(capstone_db, DB_Object.build_from_json(json_data))
        test_create(capstone_db, DB_Object.build_from_json(json_data))
        test_delete_all(capstone_db)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--host', help='Server hostname', default=HOST_NAME)
    parser.add_argument('-p', '--port', help='Server port', default=PORT_NUMBER)
    parser.add_argument('-t', '--test', help='Run DB connection tests', action='store_true')
    args = parser.parse_args()

    main(host=args.host, port=int(args.port), test=args.test)
