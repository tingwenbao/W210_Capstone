#!/usr/bin/env python3
'''
Tests Mongo DB functionality
Author: Sombiri Enwemeka

References code from Greg Bogdan
source: https://www.freelancer.com/community/articles/crud-operations-in-mongodb-using-python
'''

import sys
from db_crud import DB_CRUD
from db_object import DB_Object


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


class database_test(object):
    ''' Test operations for mongo db
    '''

    def __init__(self, host='localhost', port=27017, db=None, col=None):
        # initializing the MongoClient, this helps to
        # access the MongoDB databases and collections
        self.repository = DB_CRUD(host, port, db=db, col=col)

    def load_all_items_from_database(self):
        print("Loading all items from database:")
        db_objects = self.repository.read()
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

    def test_create(self, new_object):
        print("\n\nSaving new_object to database")
        result = self.repository.create(new_object)
        if result.acknowledged:
            new_object['_id'] = result.inserted_id
        else:
            print("[FAILED] Could not save object")
        print("new_object saved to database")
        print("Loading new_object from database")
        db_objects = self.repository.read({'_id': new_object._id})
        for p in db_objects:
            project_from_db = DB_Object.build_from_dict(p)
            print("new_object = {}".format(project_from_db.get_as_dict()))

    def test_update(self, new_object):
        print("\n\nUpdating new_object in database")
        self.repository.update(new_object)
        print("new_object updated in database")
        print("Reloading new_object from database")
        db_objects = self.repository.read({'_id': new_object._id})
        for p in db_objects:
            project_from_db = DB_Object.build_from_dict(p)
            print("new_object = {}".format(project_from_db.get_as_dict()))

    def test_delete(self, new_object):
        print("\n\nDeleting new_object from database")
        self.repository.delete(new_object)
        print("new_object deleted from database")
        print("Trying to reload new_object from database")
        db_objects = self.repository.read({'_id': new_object._id})
        found = False
        for p in db_objects:
            found = True
            project_from_db = DB_Object.build_from_dict(p)
            print("new_object = {}".format(project_from_db.get_as_dict()))

        if not found:
            print("Item with id = {} was not found in the database".format(new_object._id))

    def test_delete_all(self):
        print("\n\nDeleting EVERYTHING from database")
        self.repository.nuke()
        print("NUKED database")
        print("Trying to reload new_object from database")
        db_objects = self.repository.read()
        at_least_one_item = False
        for p in db_objects:
            at_least_one_item = True
        if at_least_one_item:
            print("[FAILED] Items still in " + self.repository.collection + " database")
        else:
            print("[SUCCESS] No items in " + self.repository.collection + " database")

    def test_db(self):
        '''
            Test database CRUD ops
        '''

        #display all items from DB
        self.load_all_items_from_database()

        #create new_object and read back from database
        json_data = {
            "title": "Wordpress website for Freelancers",
            "description": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc molestie. ",
            "price": 250,
            "assigned_to": "John Doe"}
        new_object = DB_Object.build_from_dict(json_data)
        self.test_create(new_object)

        #update new_object and read back from database
        new_object.price = 350
        self.test_update(new_object)

        #delete new_object and try to read back from database
        self.test_delete(new_object)

        #Test nuking and reading anything back from database
        for i in range(3):
            self.repository.create(DB_Object.build_from_dict(json_data))
        self.test_delete_all()
