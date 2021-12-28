class RolloutException(Exception):
    def __init__(self, message):
        super(RolloutException, self).__init__(message)