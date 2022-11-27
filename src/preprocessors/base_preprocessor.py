class BasePreprocessor:

    def __init__(self):
        self.fn = None

    def __call__(self, text :str):
        """
        Description: Applies the fn that is set by other child constructors
        """
        assert text is not None, "Text to preprocess cannot be None"
        return self.fn(text)