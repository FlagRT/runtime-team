import unittest
from compare import registrations
from relu_routes import check_model_routes


class RegistrationTests(unittest.TestCase):
    def test_only_literal_impl_names(self):
        source='lib.impl("embedding", fn, key)\nlib.impl(dynamic_name, fn, key)\n'
        self.assertEqual(registrations(source),{'embedding'})

    def test_gems_names_are_exact_overloads(self):
        source='REG=[("add.Tensor", add), ("relu", relu), ("unrelated", 3)]'
        self.assertEqual(registrations(source,True),{'add.Tensor','relu'})

    def test_relu_target_keeps_unselected_sigmoid_native(self):
        routes=[['relu','flaggems','forward','completed',3],
                ['relu','flaggems','backward','completed',3],
                ['sigmoid','native','forward','operator_disabled',1]]
        check_model_routes(routes,'candidate')
        with self.assertRaises(AssertionError):check_model_routes(routes[:-1],'on')

    def test_native_modes_require_exact_reasons(self):
        for mode,reason in [('off','experimental_disabled'),('unknown','workload_unverified')]:
            routes=[['relu','native','forward',reason,3],['sigmoid','native','forward','operator_disabled',1]]
            check_model_routes(routes,mode)
            with self.assertRaises(AssertionError):check_model_routes(routes,'on')


if __name__=='__main__':unittest.main()
