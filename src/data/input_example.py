

class InputExample(object):
    """A single example sentence"""

    def __init__(self, text, label=None):
        """Constructs a InputExample.
        Args:
          text: string. List of tokens
          label: (Optional). List of labels.
        """
        self.text = text
        self.label = label
    