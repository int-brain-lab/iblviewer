from collections import OrderedDict


class Collection(OrderedDict):

    current = None
    current_key_id = 0
    targets = []

    def get_current_key(self):
        """
        Get the current key that maps to the current object
        """
        return self.get_keys()[self.current_key_id]

    def set_current(self, key_or_id):
        """
        Set the current data
        :param key_or_id: A valid key or an int
        """
        keys = self.get_keys()
        if key_or_id in self:
            key = key_or_id
        elif isinstance(key_or_id, int):
            key_id = key_or_id
            try:
                key = keys[key_id]
            except Exception:
                return
        else:
            return
        
        key_id = 0
        for other_key in keys:
            if other_key == key:
                break
            key_id += 1
        
        # This is for when you need to know where in the array is the key
        self.current_key_id = key_id
        self.current = self.get(key)

    def next(self, loop=False):
        """
        Set the next data as current one
        :param loop: Whether looping is enabled,
        which means that if next value is out of range,
        we start back at 0
        """
        keys = self.get_keys()
        new_id = self.current_id + 1
        if new_id > len(keys) - 1:
            new_id = 0 if loop else len(keys) - 1
        self.set_current(new_id)
        return new_id

    def previous(self, loop=False):
        """
        Set the previous data as current one
        :param loop: Whether looping is enabled,
        which means that if previous value is out of range,
        we go to -1 in a backward loop
        """
        keys = self.get_keys()
        new_id = self.current_id - 1
        if new_id < 0:
            new_id = len(keys) - 1 if loop else 0
        self.set_current(new_id)
        return new_id

    def store(self, data, data_id=None, replace_existing=True, set_current=False):
        """
        Store a data
        :param data: data instance
        :param data_id: data id, a unique string if possible.
        If it's not a unique string, it will be appended a
        number so that it's unique.
        :param replace_existing: Whether any existing data with the same id is replaced or not
        :param set_current: Whether the newly stored data is set as the current one
        :return: The final data_id
        """
        if data_id is None:
            data_id = self.get_new_name(data)
        if data_id in self and not replace_existing:
            return
        self[data_id] = data
        if set_current or len(self) == 1:
            self.set_current(data_id)
        return data_id

    def get_or_create(self, data_id, data_class=None):
        """
        Get a data from a dictionary of datas or create it if none found.
        This method works only with datas that have a name (str) property.
        :param data_id: Either a data name or its id
        :param data_class: The class of the data
        :return: A data of type data_class
        """
        name = data_id
        if isinstance(data_id, int):
            name = self.get_keys()[data_id]
        data = self.get(name)
        if data is None and data_class is not None:
            if isinstance(data_id, str):
                data = data_class(name=data_id)
            else:
                data = data_class()
                data_id = self.get_new_name(data)
            self[data_id] = data
        return data

    def get_keys(self):
        """
        Get all data ids
        """
        return list(self.keys())

    def find_keys(self, id_or_subid):
        """
        Get all ids/keys that have the given param as a substring
        """
        keys = self.get_keys()
        found_ones = []
        for key in keys:
            if id_or_subid in key: #key could be an array or any iterable too
                found_ones.append(key)
        return found_ones

    def get_new_name(self, obj):
        """
        Get a new name derived from an object's class name
        :return: String
        """
        return f'{obj.__class__.__name__}{len(self)}'

    def get_name(self, *args):
        """
        Get given arguments separated by underscores as a single string
        :return: String
        """
        return '_'.join(args)
