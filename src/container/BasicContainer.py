# Template class that stores required funcs for container classes

class Container(object):
    def __init__(self, root: str, *args, **kwargs):
        # Elemental attrs -- file_path, loaded_data and datatype
        self.root = root
        self.data = None
        self.data_type = None

    def _load_data(self, **kwargs):
        raise NotImplementedError('Each subclass should emplement this itself.')


