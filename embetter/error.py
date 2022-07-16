class NotInstalled:
    """
    This object is used for optional dependencies. If a backend is not installed we
    replace the transformer/language with this object. This allows us to give a friendly
    message to the user that they need to install extra dependencies as well as a link
    to our documentation page.
    """

    def __init__(self, tool, dep):
        self.tool = tool
        self.dep = dep

        msg = f"In order to use {self.tool} you'll need to install via;\n\n"
        msg += f"pip install embetter[{self.dep}]\n\n"
        self.msg = msg

    def __getattr__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.msg)

    def __call__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.msg)
