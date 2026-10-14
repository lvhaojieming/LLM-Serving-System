import unittest
import numpy as np
import torch
from shared.ascend_int4.layout import decode, unpack_last


class AscendInt4LayoutTests(unittest.TestCase):
    def test_awq_known_nibble_order_and_zero_points(self):
        # Logical 0..7 are stored in physical order 0,2,4,6,1,3,5,7.
        packed = torch.tensor([[0x75316420]],dtype=torch.int32)
        q,z = decode(packed,packed,'awq')
        np.testing.assert_array_equal(q,[[0,1,2,3,4,5,6,7]])
        np.testing.assert_array_equal(z,[[0,1,2,3,4,5,6,7]])

    def test_legacy_gptq_packs_k_and_stores_zero_minus_one(self):
        weight = torch.tensor([[0x76543210]*8],dtype=torch.int32)
        zeros = torch.tensor([[0x77777777]],dtype=torch.int32)
        q,z = decode(weight,zeros,'gptq')
        np.testing.assert_array_equal(q,np.arange(8)[:,None].repeat(8,axis=1))
        np.testing.assert_array_equal(z,[[8]*8])

    def test_negative_packed_word_has_unsigned_nibbles(self):
        np.testing.assert_array_equal(unpack_last(torch.tensor([[-1]],dtype=torch.int32)),[[15]*8])
        with self.assertRaises(ValueError):
            unpack_last(torch.zeros(1,1,dtype=torch.float16))
