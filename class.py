
"""Simple Cluster class: use properties (no name clash between attribute and method)

This file demonstrates a safe pattern: private attributes (prefixed with
``_``) and ``@property`` accessors / setters. Also we avoid shadowing the
class name by using a different variable when instantiating.
"""

class Cluster:
    """Represent a segmented cluster (area, morphological sizes and HSV values)."""
    def __init__(self, area=0, morph_size=0, morph_size2=0, h_value=0, s_value=0, v_value=0):
        # store as private attributes to avoid name collisions with property methods
        self._area = area
        self._morph_size = morph_size
        self._morph_size2 = morph_size2
        self._h_value = h_value
        self._s_value = s_value
        self._v_value = v_value

    # area property
    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, value):
        self._area = value

    # morph_size property
    @property
    def morph_size(self):
        return self._morph_size

    @morph_size.setter
    def morph_size(self, value):
        self._morph_size = value

    @property
    def morph_size2(self):
        return self._morph_size2

    @morph_size2.setter
    def morph_size2(self, value):
        self._morph_size2 = value

    # HSV values
    @property
    def h_value(self):
        return self._h_value

    @h_value.setter
    def h_value(self, value):
        self._h_value = value

    @property
    def s_value(self):
        return self._s_value

    @s_value.setter
    def s_value(self, value):
        self._s_value = value

    @property
    def v_value(self):
        return self._v_value

    @v_value.setter
    def v_value(self, value):
        self._v_value = value


if __name__ == '__main__':
    # example usage: don't name the instance the same as the class
    val = 42
    c = Cluster()
    c.area = val   # use property assignment
    # print the attribute value (no parentheses)
    print(f"{c.area}")
