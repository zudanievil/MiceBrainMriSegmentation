"""
utility functions for text and binary processing
"""
import re
import struct

_LOC = dict()
_GLOB = dict()


def fstring_to_regex(fstring: str) -> re.Pattern:
    chunks = fstring.replace('{', '}').split('}')
    chunks = [f'(?P<{c}>.+?)' if i % 2 == 1 else c for i, c in enumerate(chunks)]
    regex = '\\b' + ''.join(chunks) + '\\b'
    return re.compile(regex)


def nifti_struct() -> struct.Struct:  # TODO: rewrite into smth humanly readable
    # c-struct info: https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
    s = ('<',
         'i',  # int  sizeof_hdr     == 348
         '10s',  # char data_type[10] (unused)
         '18s',  # char db_name[18]   (unused)
         'i',  # int  extents       (unused)
         'h',  # short session_error(unused)
         'c',  # char  regular      (unused)

         'c',  # char  dim_info
         '8h',  # short dim[8]       #dim[0] -- num of dims, in range(1,8), dim[i>0] -- dim size;
         # dim[1-3] - {x, y, z}, dim[4] - time, dim[5-7] -- something else
         'f',  # float intent_p1    #what to do with dim 5
         'f',  # float intent_p2    #what to do with dim 6
         'f',  # float intent_p3    #what to do with dim 7
         'h',  # short intent_code  #how to interpret dims 5-7
         'h',  # short datatype     #
         'h',  # short bitpix       #voxel bItsize (bytesize*8)
         'h',  # short slice_start
         '8f',  # float pixdim[8]    #voxel length among axis i
         'f',
         # float vox_offset   #byte offset of image data (0 for .img; for .nii multiple of 16, >=352) 24-mantissa float
         'f',
         # float scl_slope    # v' = scl_slope * v + scl_inter if scl_slope != 0.0 (v'--actual voxel value, v--stored value)
         'f',  # float scl_inter
         'h',  # short slice_end
         'c',  # char  slice_code
         'f',  # float cal_max      #'white' voxel value
         'f',  # float cal_min      #'black' voxel value
         'f',  # float slice_duration
         'f',  # float toffset
         'i',  # int   glmax
         'i',  # int   glmin
         '80s',  # char  descrip[80]  #some string
         '24s',  # char  aux_file[24]
         'h',
         # short qform_code #coordinate system type, 0 -- analyze7.5-like, >0 -- nifti type, use qaternion, offset and pixdim
         'h',  # short sform_code #coordinate system, if >0, use srow matrix
         'f',  # float quatern_b
         'f',  # float quatern_c
         'f',  # float qoffset_d
         'f',  # float qoffset_x
         'f',  # float qoffset_y
         'f',  # float qoffset_z
         '4f',  # float srow_x[4]
         '4f',  # float srow_y[4]
         '4f',  # float srow_z[4]
         '16s',  # char intent_name[16]
         '4s',  # char magic[4]
         )
    s = ' '.join(s)
    return struct.Struct(s)
