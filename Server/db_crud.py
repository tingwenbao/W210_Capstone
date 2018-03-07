from pymongo import MongoClient
from bson.objectid import ObjectId
from db_object import DB_Object


class DB_CRUD(object):
    """ Repository implementing CRUD operations on objects in MongoDB """

    def __init__(self, host='localhost', port=27017, db=None, col=None):
        # initializing the MongoClient, this helps to
        # access the MongoDB databases and collections
        if db is None or col is None:
            raise Exception("Database and collection names are required")
        self.client = MongoClient(host=host, port=port)
        self.database = self.client[db]
        self.collection = col

    def create(self, db_object):
        if db_object is not None:
            self.database[self.collection].insert(db_object.get_as_json())
        else:
            raise Exception("Nothing to save, because db_object parameter is None")

    def read(self, _id=None):
        if _id is None:
            return self.database[self.collection].find({})
        else:
            return self.database[self.collection].find({"_id": _id})

    def update(self, db_object):
        if db_object is not None:
            # the save() method updates the document if this has an _id property
            # which appears in the collection, otherwise it saves the data
            # as a new document in the collection
            self.database[self.collection].save(db_object.get_as_json())
        else:
            raise Exception("Nothing to update, because db_object parameter is None")

    def delete(self, db_object):
        if db_object is not None:
            self.database[self.collection].remove(db_object.get_as_json())
        else:
            raise Exception("Nothing to delete, because db_object parameter is None")

    def nuke(self):
        self.database[self.collection].delete_many({})
