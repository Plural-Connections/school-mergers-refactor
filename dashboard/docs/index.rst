School Mergers (`eat.py`)
=========================

.. currentmodule:: eat

`eat.py` contains the definitions for the dataclasses—representing schools, districts, predicted policy impacts, and more—and the functions for "digesting" the low-level data in the `./data` directory into the dataclasses for high level analysis.

For the other code files, it's better just to read through them.

Type aliases or enums
---------------------

.. autoclass:: DemoType

.. autoenum:: StatusCode

Data classes
------------

.. autoclass:: Simulation
   :members:

.. autoclass:: District
   :members:

.. autoclass:: School
   :members:

.. autoclass:: Analytics
   :members:

.. autoclass:: Population
   :members:

.. autoclass:: TravelTimes
   :members:

.. autoclass:: Impact
   :members:
