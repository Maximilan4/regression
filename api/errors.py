

class ApiException(Exception):

    def __init__(self, message: str, *args, **kwargs):
        self.message = message
        super().__init__(args, kwargs)

    def __str__(self):
        return self.message


class NotValidRequest(ApiException):

    def __init__(self, message: str):
        message = "Request is not valid: {}".format(message)
        super().__init__(message)
