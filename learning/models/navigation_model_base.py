from learning.models.navigation_model_component_base import NavigationModelComponentBase
from learning.modules.generic_model_state import GenericModelState


class NavigationModelBase(NavigationModelComponentBase):

    def __init__(self, run_name="", domain="sim"):
        super(NavigationModelBase, self).__init__(run_name=run_name, domain=domain)
        self.set_model_state(GenericModelState())

    def steal_cross_domain_modules(self, other_self):
        self.iter = other_self.iter

