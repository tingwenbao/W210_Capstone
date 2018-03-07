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
        '''
            Creates a new object within the database
            args:
                db_object - Mongo DB object to insert
            returns:
                WriteResult - object describing the result of this operations
            Raises:
                - Duplicate key error if db_object _id is not unique
                - Exception if db_object is None
        '''
        if db_object is not None:
            return self.database[self.collection].insert(db_object.get_as_json())
        else:
            raise Exception("Nothing to save, because db_object parameter is None")

    def read(self, **kwargs):
        '''
            Read an object from the database, if '_id' is None return all DB entries
            args:
                keys to filter within DB
            returns:
                    A cursor to the documents that match the query criteria
        '''
        filter_dict = {}
        _id = kwargs.get('_id', None)
        for (k, v) in kwargs.items():
            if k == '_id':
                continue
            filter_dict[k] = v
        if _id is None:
            return self.database[self.collection].find({})
        else:
            return self.database[self.collection].find(filter_dict)

    def update(self, db_object):
        '''
            Updates an existing object within the database
            args:
                db_object - Mongo DB object to update
            returns:
                WriteResult - object describing the result of this operations
            Raises:
                - Exception if db_object is None
        '''
        if db_object is not None:
            # the save() method updates the document if this has an _id property
            # which appears in the collection, otherwise it saves the data
            # as a new document in the collection
            return self.database[self.collection].save(db_object.get_as_json())
        else:
            raise Exception("Nothing to update, because db_object parameter is None")

    def delete(self, db_object):
        '''
            Deletes an object from the database
            args:
                db_object - Mongo DB object to delete
            returns:
                WriteResult - object describing the result of this operations
            Raises:
                - Exception if db_object is None
        '''
        if db_object is not None:
            return self.database[self.collection].remove(db_object.get_as_json())
        else:
            raise Exception("Nothing to delete, because db_object parameter is None")

    def nuke(self):
        '''
            Deletes all objectes from the database collection
            returns:
                WriteResult - object describing the result of this operations
        '''
        return self.database[self.collection].delete_many({})
