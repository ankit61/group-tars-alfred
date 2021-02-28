class Config():

    static_members = {}

    def __init__(self):
        self.__dict__.update(Config.static_members)

    def __setattr__(self, name, value):
        # inspired by http://code.activestate.com/recipes/65207-constants-in-python/
        if name in self.__dict__:
            raise Exception(f'Value of {name} is already set - change original value')
        self.__dict__[name] = value

    def set_static_val(self, name, val):
        if name in Config.static_members:
            return

        Config.static_members[name] = val
        self.__dict__[name] = Config.static_members[name]

    def set_static_from_func(self, name, func, *args, **kwargs):
        if name in Config.static_members:
            return

        Config.static_members[name] = func(*args, **kwargs)

        self.__dict__[name] = Config.static_members[name]
