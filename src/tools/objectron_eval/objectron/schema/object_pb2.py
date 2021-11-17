# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor.FileDescriptor(
    name='object.proto',
    package='xeno.pursuit.proto',
    syntax='proto3',
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x0cobject.proto\x12\x12xeno.pursuit.proto\"d\n\x08KeyPoint\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\x12\x19\n\x11\x63onfidence_radius\x18\x04 \x01(\x02\x12\x0c\n\x04name\x18\x05 \x01(\t\x12\x0e\n\x06hidden\x18\x06 \x01(\x08\"\xf5\x02\n\x06Object\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x10\n\x08\x63\x61tegory\x18\x02 \x01(\t\x12-\n\x04type\x18\x03 \x01(\x0e\x32\x1f.xeno.pursuit.proto.Object.Type\x12\x10\n\x08rotation\x18\x04 \x03(\x02\x12\x13\n\x0btranslation\x18\x05 \x03(\x02\x12\r\n\x05scale\x18\x06 \x03(\x02\x12/\n\tkeypoints\x18\x07 \x03(\x0b\x32\x1c.xeno.pursuit.proto.KeyPoint\x12\x31\n\x06method\x18\x08 \x01(\x0e\x32!.xeno.pursuit.proto.Object.Method\"D\n\x04Type\x12\x12\n\x0eUNDEFINED_TYPE\x10\x00\x12\x10\n\x0c\x42OUNDING_BOX\x10\x01\x12\x0c\n\x08SKELETON\x10\x02\x12\x08\n\x04MESH\x10\x03\">\n\x06Method\x12\x12\n\x0eUNKNOWN_METHOD\x10\x00\x12\x0e\n\nANNOTATION\x10\x01\x12\x10\n\x0c\x41UGMENTATION\x10\x02\"$\n\x04\x45\x64ge\x12\x0e\n\x06source\x18\x01 \x01(\x05\x12\x0c\n\x04sink\x18\x02 \x01(\x05\"\x92\x01\n\x08Skeleton\x12\x1a\n\x12reference_keypoint\x18\x01 \x01(\x05\x12\x10\n\x08\x63\x61tegory\x18\x02 \x01(\t\x12/\n\tkeypoints\x18\x03 \x03(\x0b\x32\x1c.xeno.pursuit.proto.KeyPoint\x12\'\n\x05\x65\x64ges\x18\x04 \x03(\x0b\x32\x18.xeno.pursuit.proto.Edge\"9\n\tSkeletons\x12,\n\x06object\x18\x01 \x03(\x0b\x32\x1c.xeno.pursuit.proto.Skeletonb\x06proto3'
)

_OBJECT_TYPE = _descriptor.EnumDescriptor(
    name='Type',
    full_name='xeno.pursuit.proto.Object.Type',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name='UNDEFINED_TYPE', index=0, number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key),
        _descriptor.EnumValueDescriptor(
            name='BOUNDING_BOX', index=1, number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key),
        _descriptor.EnumValueDescriptor(
            name='SKELETON', index=2, number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key),
        _descriptor.EnumValueDescriptor(
            name='MESH', index=3, number=3,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=380,
    serialized_end=448,
)
_sym_db.RegisterEnumDescriptor(_OBJECT_TYPE)

_OBJECT_METHOD = _descriptor.EnumDescriptor(
    name='Method',
    full_name='xeno.pursuit.proto.Object.Method',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name='UNKNOWN_METHOD', index=0, number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key),
        _descriptor.EnumValueDescriptor(
            name='ANNOTATION', index=1, number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key),
        _descriptor.EnumValueDescriptor(
            name='AUGMENTATION', index=2, number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=450,
    serialized_end=512,
)
_sym_db.RegisterEnumDescriptor(_OBJECT_METHOD)

_KEYPOINT = _descriptor.Descriptor(
    name='KeyPoint',
    full_name='xeno.pursuit.proto.KeyPoint',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='x', full_name='xeno.pursuit.proto.KeyPoint.x', index=0,
            number=1, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='y', full_name='xeno.pursuit.proto.KeyPoint.y', index=1,
            number=2, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='z', full_name='xeno.pursuit.proto.KeyPoint.z', index=2,
            number=3, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='confidence_radius', full_name='xeno.pursuit.proto.KeyPoint.confidence_radius', index=3,
            number=4, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='name', full_name='xeno.pursuit.proto.KeyPoint.name', index=4,
            number=5, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='hidden', full_name='xeno.pursuit.proto.KeyPoint.hidden', index=5,
            number=6, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=36,
    serialized_end=136,
)

_OBJECT = _descriptor.Descriptor(
    name='Object',
    full_name='xeno.pursuit.proto.Object',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='id', full_name='xeno.pursuit.proto.Object.id', index=0,
            number=1, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='category', full_name='xeno.pursuit.proto.Object.category', index=1,
            number=2, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='type', full_name='xeno.pursuit.proto.Object.type', index=2,
            number=3, type=14, cpp_type=8, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='rotation', full_name='xeno.pursuit.proto.Object.rotation', index=3,
            number=4, type=2, cpp_type=6, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='translation', full_name='xeno.pursuit.proto.Object.translation', index=4,
            number=5, type=2, cpp_type=6, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='scale', full_name='xeno.pursuit.proto.Object.scale', index=5,
            number=6, type=2, cpp_type=6, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='keypoints', full_name='xeno.pursuit.proto.Object.keypoints', index=6,
            number=7, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='method', full_name='xeno.pursuit.proto.Object.method', index=7,
            number=8, type=14, cpp_type=8, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
        _OBJECT_TYPE,
        _OBJECT_METHOD,
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=139,
    serialized_end=512,
)

_EDGE = _descriptor.Descriptor(
    name='Edge',
    full_name='xeno.pursuit.proto.Edge',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='source', full_name='xeno.pursuit.proto.Edge.source', index=0,
            number=1, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='sink', full_name='xeno.pursuit.proto.Edge.sink', index=1,
            number=2, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=514,
    serialized_end=550,
)

_SKELETON = _descriptor.Descriptor(
    name='Skeleton',
    full_name='xeno.pursuit.proto.Skeleton',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='reference_keypoint', full_name='xeno.pursuit.proto.Skeleton.reference_keypoint', index=0,
            number=1, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='category', full_name='xeno.pursuit.proto.Skeleton.category', index=1,
            number=2, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='keypoints', full_name='xeno.pursuit.proto.Skeleton.keypoints', index=2,
            number=3, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='edges', full_name='xeno.pursuit.proto.Skeleton.edges', index=3,
            number=4, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=553,
    serialized_end=699,
)

_SKELETONS = _descriptor.Descriptor(
    name='Skeletons',
    full_name='xeno.pursuit.proto.Skeletons',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='object', full_name='xeno.pursuit.proto.Skeletons.object', index=0,
            number=1, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=701,
    serialized_end=758,
)

_OBJECT.fields_by_name['type'].enum_type = _OBJECT_TYPE
_OBJECT.fields_by_name['keypoints'].message_type = _KEYPOINT
_OBJECT.fields_by_name['method'].enum_type = _OBJECT_METHOD
_OBJECT_TYPE.containing_type = _OBJECT
_OBJECT_METHOD.containing_type = _OBJECT
_SKELETON.fields_by_name['keypoints'].message_type = _KEYPOINT
_SKELETON.fields_by_name['edges'].message_type = _EDGE
_SKELETONS.fields_by_name['object'].message_type = _SKELETON
DESCRIPTOR.message_types_by_name['KeyPoint'] = _KEYPOINT
DESCRIPTOR.message_types_by_name['Object'] = _OBJECT
DESCRIPTOR.message_types_by_name['Edge'] = _EDGE
DESCRIPTOR.message_types_by_name['Skeleton'] = _SKELETON
DESCRIPTOR.message_types_by_name['Skeletons'] = _SKELETONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

KeyPoint = _reflection.GeneratedProtocolMessageType('KeyPoint', (_message.Message,), {
    'DESCRIPTOR': _KEYPOINT,
    '__module__': 'object_pb2'
    # @@protoc_insertion_point(class_scope:xeno.pursuit.proto.KeyPoint)
})
_sym_db.RegisterMessage(KeyPoint)

Object = _reflection.GeneratedProtocolMessageType('Object', (_message.Message,), {
    'DESCRIPTOR': _OBJECT,
    '__module__': 'object_pb2'
    # @@protoc_insertion_point(class_scope:xeno.pursuit.proto.Object)
})
_sym_db.RegisterMessage(Object)

Edge = _reflection.GeneratedProtocolMessageType('Edge', (_message.Message,), {
    'DESCRIPTOR': _EDGE,
    '__module__': 'object_pb2'
    # @@protoc_insertion_point(class_scope:xeno.pursuit.proto.Edge)
})
_sym_db.RegisterMessage(Edge)

Skeleton = _reflection.GeneratedProtocolMessageType('Skeleton', (_message.Message,), {
    'DESCRIPTOR': _SKELETON,
    '__module__': 'object_pb2'
    # @@protoc_insertion_point(class_scope:xeno.pursuit.proto.Skeleton)
})
_sym_db.RegisterMessage(Skeleton)

Skeletons = _reflection.GeneratedProtocolMessageType('Skeletons', (_message.Message,), {
    'DESCRIPTOR': _SKELETONS,
    '__module__': 'object_pb2'
    # @@protoc_insertion_point(class_scope:xeno.pursuit.proto.Skeletons)
})
_sym_db.RegisterMessage(Skeletons)

# @@protoc_insertion_point(module_scope)
