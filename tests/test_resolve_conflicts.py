import unittest

from src.matching.resolve_conflicts import confidence_gap,selected_owners


class OwnershipTests(unittest.TestCase):
    def test_one_query_can_own_many_targets_but_each_target_has_one_owner(self):
        rows=[('q1','t1',.99,1,.95,2),('q2','t1',.95,2,None,2),
              ('q1','t2',.98,1,None,1)]
        self.assertEqual(selected_owners(rows,0),{'t1':'q1','t2':'q1'})

    def test_exact_ties_abstain_instead_of_using_id_order_as_evidence(self):
        rows=[('q1','t',.99,1,.99,2),('q2','t',.99,2,None,2)]
        self.assertEqual(selected_owners(rows,0),{})

    def test_margin_is_applied_only_to_competing_owners(self):
        rows=[('q1','t1',.991,1,.99,2),('q1','t2',.92,1,None,1)]
        self.assertEqual(selected_owners(rows,2),{'t2':'q1'})
        self.assertGreater(confidence_gap(.99,.9),2)


if __name__=='__main__':
    unittest.main()
