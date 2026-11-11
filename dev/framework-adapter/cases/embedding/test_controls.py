import unittest
from admission import check_model_routes, OPERATORS

class CaseTests(unittest.TestCase):
    def test_on_requires_both_embeddings_both_stages(self):
        routes=[['embedding','flaggems','forward','completed',2],
                ['embedding','flaggems','backward','completed',2]]
        check_model_routes(routes,'on');check_model_routes(routes,'candidate')
        with self.assertRaises(AssertionError):check_model_routes(routes[:1],'on')
        routes[1][-1]=1
        with self.assertRaises(AssertionError):check_model_routes(routes,'on')
    def test_native_controls_zero_target(self):
        for mode,reason in [('off','experimental_disabled'),('unknown','workload_unverified')]:
            check_model_routes([['embedding','native','forward',reason,2]],mode)
            with self.assertRaises(AssertionError):check_model_routes([],mode)
        check_model_routes([],'native');check_model_routes([],'native_repeat')
    def test_identity_includes_indices_and_options_contract(self):
        spec=OPERATORS['embedding']
        self.assertEqual(spec['indices_shape'],[32,5])
        self.assertEqual(spec['indices_dtype'],'torch.int64')
        self.assertEqual(spec['contract'],'embedding-dense-no-padding-v1')
        self.assertEqual(spec['weight_shapes'],[[3850,16],[3850,1]])

if __name__=='__main__':unittest.main()
