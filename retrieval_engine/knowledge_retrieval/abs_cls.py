
class SingletonMeta(type):
    """
    Metaclass for ensuring a class has only one instance and providing a global point of access to it.
    """
    _instances = {} 

    def __call__(cls, *args, **kwargs):
        """
        This method is called when a class is instantiated.
        It checks if an instance already exists and returns it if so.
        Otherwise, it creates a new instance and stores it in the _instances dictionary.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    