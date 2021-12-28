
class AbstractPolicy():

    def start_sequence(self):
        ...

    def get_action(self, state, instruction=None):
        ...
