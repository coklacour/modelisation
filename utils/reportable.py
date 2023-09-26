#############################################################################
#                                 Packages                                  #
#############################################################################

from datapane import Group, Text
from abc import ABCMeta, abstractmethod

#############################################################################
#                                 Scripts                                   #
#############################################################################


class ReportableAbstract(metaclass=ABCMeta):
    """
    Define a basic structure for creating report blocks from certain steps in modeling pipelines.
    """

    GROUP_NAME = "Block default name"

    def __init__(self, *args, **kwargs):
        """
        Instanciate the Reportable interface in a cooperative-inheritance way
        """

        super().__init__(*args, **kwargs)

    @property
    def group_name(self):
        return self.GROUP_NAME

    @abstractmethod
    def get_report_group(self) -> Group:
        """
        Extract the report group

        Returns:
            Group: The group to be wrapped in a report.
        """

        raise NotImplementedError("Must be implemented in concrete classe.")
