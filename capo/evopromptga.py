"""
Wrapper for EvoPromptGA that provides picklable functionality.
Contains the EvoPromptGAPickable class which inherits from EvoPromptGA and enables serialization for distributed computing.
"""

from logging import Logger, getLogger

from promptolution.optimizers.evoprompt_ga import EvoPromptGA

logger = Logger(__name__)


class EvoPromptGAPickable(EvoPromptGA):
    """Inherit from EvoPromptGA and make it pickable, and create attribute for downstream LLM."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downstream_llm = self.predictor.llm

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        state.pop("predictor", None)
        state.pop("logger", None)
        state.pop("meta_llm", None)
        state.pop("downstream_llm", None)

        return state

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__dict__.update(state)
        self.predictor = None
        self.logger = getLogger(__name__)
