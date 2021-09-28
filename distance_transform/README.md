# README

## Prerequisites
Each DT algorithm requires a gmap as input.
The gmap has to have the following interface:

``` 
def get_dart_by_indetifier(identifier: int) -> Dart:
    """
        It returs the dart given the identifier
    """

def ai(i: int, index: int) -> Dart:
    """
        It the returns, if it exist, the Dart associated with
        "index" according to alfai
    """
    
def n() -> int:
    """
        It returs the dimension of the gmap
    """
    
def darts_with_attributes() -> iterator<Dart>:
    """
        It returns an iterator on darts
    """
```