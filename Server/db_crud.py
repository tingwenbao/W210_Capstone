from pymongo import MongoClient
import gridfs
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
            return self.database[self.collection].insert_one(db_object.get_as_dict())
        else:
            raise Exception("Nothing to save, because db_object parameter is None")

    def read(self, filter_dict={}, **kwargs):
        '''
            Read an object from the database, if '_id' is None return all DB entries
            args:
                keys to filter within DB
            returns:
                    A cursor to the documents that match the query criteria
            Additional info:
            http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.find
        '''
        return self.database[self.collection].find(filter_dict, **kwargs)

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
            return self.database[self.collection].save(db_object.get_as_dict())
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
            return self.database[self.collection].remove(db_object.get_as_dict())
        else:
            raise Exception("Nothing to delete, because db_object parameter is None")

    def nuke(self):
        '''
            Deletes all objectes from the database collection
            returns:
                WriteResult - object describing the result of this operations
        '''
        return self.database[self.collection].drop()

    def createIndex(self, fields, **kwargs):
        '''
            Creates searchable index for input fields
        '''
        self.database[self.collection].create_index(fields, **kwargs)

    def stats(self):
        '''
            Returns dictionary of collection statistics
        '''
        # Check if collection exists in DB
        if self.collection in self.database.collection_names():
            return self.database.command("collstats", self.collection)
        else:
            return {'info': "'" + self.collection + "' doesn't exist in database"}

    def create_file(self, data, **kwargs):
        '''
            Uses gridfs to store files in the database.
            Useful for storing objects which are larger than the 16MB document size limit
        '''
        fs = gridfs.GridFS(self.database)
        return fs.put(data, **kwargs)

    def read_file(self, filename, **kwargs):
        '''
            Uses gridfs to retrieve a file from the database.
        '''
        fs = gridfs.GridFS(self.database)
        return fs.find({"filename": filename}, **kwargs)
