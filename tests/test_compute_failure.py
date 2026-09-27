import unittest
from src.compute import failure_hint

class ComputeFailureTests(unittest.TestCase):
    def test_access_denied_has_launch_guidance(self):
        for reason in ('アクセスが拒否されました', 'Access is denied'):
            text=failure_hint('ImportError: DLL load failed: '+reason)
            self.assertIn('start.ps1',text)
            self.assertIn('推論開始前',text)

    def test_unrelated_failure_does_not_claim_access_denied(self):
        self.assertNotIn('アクセスが拒否',failure_hint('CUDA out of memory'))
