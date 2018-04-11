from bson.objectid import ObjectId
import json
from pprint import pformat


class DB_Object(object):
    """A class for storing DB_Object related information"""

    def __init__(self, **kwargs):
        if kwargs:
            for (k, v) in kwargs.items():
                super().__setattr__(k, v)

    def get_as_dict(self):
        """ Method returns a dict representing  the DB_Object object,
         this can be written to a JSON file or saved to MongoDB """
        return self.__dict__.copy()

    def get(self, key, def_val):
        return self.__dict__.get(key, def_val)

    def pop(self, key):
        return self.__dict__.pop(key)

    def items(self):
        return self.__dict__.items()

    @staticmethod
    def build_from_dict(json_data):
        """ Method used to build DB_Object objects from JSON data returned from MongoDB
        (stored as python dict) """
        if json_data is not None:
            try:
                return DB_Object(**json_data)
            except KeyError as e:
                raise Exception("Key not found in json_data: {}".format(e.message))
        else:
            raise Exception("No data to create DB_Object from!")

    def __getitem__(self, k):
        return self.__getattribute__(k)

    def __setitem__(self, k, v):
        if k == '_id':
            super().__setattr__(k, v)
        else:
            raise KeyError("DB_Object only accepts setting of '_id' attribute")

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        elif isinstance(o, DB_Object):
            o = o.get_as_dict()
            o['_id'] = str(o.get('_id', None))
            if 'acne_products' in o:
                o['acne_products'] = [str(obj_id) for obj_id in o['acne_products']]

            # Convert floats to strings
            for (k, v) in o.items():
                if type(v) is float:
                    o[k] = str(v)
            return o
        return json.JSONEncoder.default(self, o)
