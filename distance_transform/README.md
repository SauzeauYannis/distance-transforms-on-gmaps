# README

## Prerequisites
Each DT algorithm requires a gmap as input.
The gmap has to have the following interface:

```python
def get_dart_by_indetifier(identifier: int) -> Dart:
    """
        It returns the dart given the identifier
    """

def ai(i: int, index: int) -> Dart:
    """
        It the returns, if it exist, the Dart associated with
        "index" according to alfa_i
    """
    
def n() -> int:
    """
        It returns the dimension of the gmap
    """
    
def darts_with_attributes() -> iterator<Dart>:
    """
        It returns an iterator on darts
    """
```