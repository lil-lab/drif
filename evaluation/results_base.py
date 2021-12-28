
class EvaluationResults:

    def __init__(self):
        self.state = {}

    def __add__(self, past_results):
        return self

    def get_dict(self):
        return self.state

    def set_dict(self, dict):
        self.state = dict