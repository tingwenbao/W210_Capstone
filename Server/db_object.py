from bson.objectid import ObjectId


class DB_Object(object):
    """A class for storing DB_Object related information"""

    def __init__(self, **kwargs):
        if kwargs:
            self._id = kwargs.get('_id', ObjectId())
            for (k, v) in kwargs.items():
                if k == '_id':
                    continue
                super().__setattr__(k, v)

    def get_as_json(self):
        """ Method returns the JSON representation of the DB_Object object, which can be saved to MongoDB """
        return self.__dict__

    @staticmethod
    def build_from_json(json_data):
        """ Method used to build DB_Object objects from JSON data returned from MongoDB """
        if json_data is not None:
            try:
                return DB_Object(**json_data)
            except KeyError as e:
                raise Exception("Key not found in json_data: {}".format(e.message))
        else:
            raise Exception("No data to create DB_Object from!")
