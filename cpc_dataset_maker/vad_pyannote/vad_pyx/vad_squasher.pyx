import torch
import numpy as np
from cpython cimport bool
cimport numpy as np
cimport cython
ctypedef np.float32_t CTYPE_t # cost type
ctypedef np.intp_t IND_t # array index type
CTYPE = np.float32 # cost type



def get_intervals_from_proba(vad_vector,
                             time_chunk,
                             offset_time_chunk,
                             onset, offset,
                             pad_start, pad_end):


    size = vad_vector.size(0)
    output, n_items = _make_vad_vector(size, time_chunk, offset_time_chunk,
                            vad_vector.numpy(), onset, offset,
                            pad_start, pad_end)

    return list(output[:n_items])

def remove_short_seq(vad_segments, min_size):

    size = len(vad_segments)
    in_array = np.asarray(vad_segments).astype(np.float32)
    output, n_items = _remove_short_seq(in_array,
                                        size,
                                        float(min_size))
    output = output[:n_items]
    return list(output)


def merge_short_voids(vad_segments, min_size):

    size = len(vad_segments)
    in_array = np.asarray(vad_segments).astype(np.float32)
    output, n_items = _merge_short_voids(in_array,
                                         size,
                                         float(min_size))
    output = output[:n_items]
    return list(output)


def build_vad_intervals(vad_vector,
                        time_chunk,
                        onset,
                        offset,
                        offset_time_chunk=0,
                        pad_start=0,
                        pad_end=0,
                        min_size_sil=0,
                        min_size_voice=0):

    size = vad_vector.size(0)
    if size == 0:
      return []
    vad_intervals, n_items = _make_vad_vector(size, time_chunk, offset_time_chunk,
                                              vad_vector.numpy(), onset, offset,
                                              pad_start, pad_end)

    if min_size_voice > 0:
        vad_intervals, n_items = _remove_short_seq(vad_intervals,
                                                   n_items,
                                                   min_size_voice)

    if min_size_sil > 0:
        vad_intervals, n_items = _merge_short_voids(vad_intervals,
                                                    n_items,
                                                    min_size_sil)
    return np.asarray(vad_intervals[:n_items]).tolist()


cpdef _make_vad_vector(IND_t      size,
                       CTYPE_t    time_chunk,
                       CTYPE_t    offset_time_chunk,
                       CTYPE_t[:] proba_array,
                       CTYPE_t    onset,
                       CTYPE_t    offset,
                       CTYPE_t    pad_start,
                       CTYPE_t    pad_end):

  cdef CTYPE_t[:,:] output = np.empty((size, 2), dtype=CTYPE)
  cdef IND_t n_items = 0
  cdef CTYPE_t curr_end

  cdef bool status = proba_array[0] > onset
  cdef CTYPE_t start = offset_time_chunk

  for index in range(1, size):
    vad = proba_array[index]
    if status:
      if vad < offset:
        curr_end = index * time_chunk + pad_end + offset_time_chunk
        if n_items > 0 and start <= output[n_items - 1][1]:
          output[n_items - 1][1] = curr_end
        else:
          output[n_items][0] = start
          output[n_items][1] = curr_end
          n_items+=1
        status = False

    # currently inactive
    else:
      # switching from inactive to active
      if vad > onset:
        start = index * time_chunk - pad_start + offset_time_chunk
        status = True

  if status:
    curr_end = size * time_chunk + offset_time_chunk
    if n_items > 0 and start <= output[n_items - 1][1]:
      output[n_items - 1][1] = curr_end
    else:
      output[n_items][0] = start
      output[n_items][1] = curr_end
      n_items+=1

  return output, n_items


cpdef _remove_short_seq(CTYPE_t[:,:] vad_segments,
                        IND_t        n_segments,
                        CTYPE_t      min_size):

  cdef CTYPE_t[:,:] output = np.empty((n_segments, 2), dtype=CTYPE)
  cdef CTYPE_t start
  cdef CTYPE_t end

  cdef IND_t n_items = 0

  for index in range(n_segments):
    start = vad_segments[index][0]
    end = vad_segments[index][1]
    if end - start > min_size:
      output[n_items][0] = start
      output[n_items][1] = end
      n_items+=1

  return output, n_items

cpdef _merge_short_voids(CTYPE_t[:,:] vad_segments,
                         IND_t        n_segments,
                         CTYPE_t      min_size):

  cdef CTYPE_t[:,:] output = np.empty((n_segments, 2), dtype=CTYPE)
  cdef CTYPE_t start
  cdef CTYPE_t end
  cdef CTYPE_t last_void_start = 0
  cdef IND_t n_items = 0

  for index in range(n_segments):
    start = vad_segments[index][0]
    end = vad_segments[index][1]
    if start - last_void_start < min_size:
      if n_items == 0:
        output[0, 0] = last_void_start
        output[0, 1] = end
        n_items+=1
      else:
        output[n_items-1, 1] = end
    else:
      output[n_items, 0] = start
      output[n_items, 1] = end
      n_items+=1
    last_void_start = end

  return output, n_items
