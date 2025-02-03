#
#  Copyright 2019 Mikko Korpela
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import sys
import dis
from typing import List, Tuple, TypeVar
from types import FrameType, FunctionType

__VERSION__ = "3.1.0"


_WrappedMethod = TypeVar("_WrappedMethod", bound=FunctionType)


def overrides(method: _WrappedMethod) -> _WrappedMethod:
    """Decorator to indicate that the decorated method overrides a method in
    superclass.
    The decorator code is executed while loading class. Using this method
    should have minimal runtime performance implications.

    This is based on my idea about how to do this and fwc:s highly improved
    algorithm for the implementation fwc:s
    algorithm : http://stackoverflow.com/a/14631397/308189
    my answer : http://stackoverflow.com/a/8313042/308189

    How to use:
    from overrides_ import overrides

    class SuperClass(object):
        def method(self):
          return 2

    class SubClass(SuperClass):

        @overrides
        def method(self):
            return 1

    :raises  AssertionError if no match in super classes for the method name
    :return  method with possibly added (if the method doesn't have one)
        docstring from super class
    """
    setattr(method, "__override__", True)
    for super_class in _get_base_classes(sys._getframe(2), method.__globals__):
        if hasattr(super_class, method.__name__):
            super_method = getattr(super_class, method.__name__)
            if hasattr(super_method, "__finalized__"):
                finalized = getattr(super_method, "__finalized__")
                if finalized:
                    raise AssertionError('Method "%s" is finalized' % method.__name__)
            if not method.__doc__:
                method.__doc__ = super_method.__doc__
            return method
    raise AssertionError('No super class method found for "%s"' % method.__name__)


def _get_base_classes(frame, namespace):
    return [
        _get_base_class(class_name_components, namespace)
        for class_name_components in _get_base_class_names(frame)
    ]


def op_stream(code, max):
    """Generator function: convert Python bytecode into a sequence of
    opcode-argument pairs."""
    i = [0]

    def next():
        val = code[i[0]]
        i[0] += 1
        return val

    ext_arg = 0
    while i[0] <= max:
        op, arg = next(), next()
        if op == dis.EXTENDED_ARG:
            ext_arg += arg
            ext_arg <<= 8
            continue
        else:
            yield (op, arg + ext_arg)
            ext_arg = 0


def _get_base_class_names(frame: FrameType) -> List[List[str]]:
    """Get baseclass names from the code object"""
    current_item: List[str] = []
    items: List[List[str]] = []
    add_last_step = True

    for instruction in dis.get_instructions(frame.f_code):
        if instruction.offset > frame.f_lasti:
            break
        if instruction.opcode not in dis.hasname:
            continue
        if not add_last_step:
            items = []
            add_last_step = True

        # Combine LOAD_NAME and LOAD_GLOBAL as they have similar functionality
        if instruction.opname in ["LOAD_NAME", "LOAD_GLOBAL"]:
            if current_item:
                items.append(current_item)
            current_item = [instruction.argval]

        elif instruction.opname == "LOAD_ATTR" and current_item:
            current_item.append(instruction.argval)

        # Reset on other instructions
        else:
            if current_item:
                items.append(current_item)
            current_item = []
            add_last_step = False

    if current_item:
        items.append(current_item)
    return items


def _get_base_class(components, namespace):
    try:
        obj = namespace[components[0]]
    except KeyError:
        if isinstance(namespace["__builtins__"], dict):
            obj = namespace["__builtins__"][components[0]]
        else:
            obj = getattr(namespace["__builtins__"], components[0])
    for component in components[1:]:
        if hasattr(obj, component):
            obj = getattr(obj, component)
    return obj
