import unittest

from youtube_live import dashboard_percentile, dashboard_request_authorized


class DashboardTests(unittest.TestCase):
    def test_dashboard_requires_matching_bearer_token_when_configured(self) -> None:
        self.assertTrue(
            dashboard_request_authorized("secret", "Bearer secret", "203.0.113.10")
        )
        self.assertFalse(
            dashboard_request_authorized("secret", "Bearer wrong", "127.0.0.1")
        )
        self.assertFalse(
            dashboard_request_authorized("secret", "", "127.0.0.1")
        )

    def test_dashboard_without_token_is_localhost_only(self) -> None:
        self.assertTrue(dashboard_request_authorized("", "", "127.0.0.1"))
        self.assertTrue(dashboard_request_authorized("", "", "::1"))
        self.assertFalse(dashboard_request_authorized("", "", "203.0.113.10"))

    def test_dashboard_percentile_uses_nearest_rank(self) -> None:
        self.assertEqual(dashboard_percentile([10, 20, 30, 40, 50], 0.95), 50.0)
        self.assertEqual(dashboard_percentile([50, 10, 30, 20, 40], 0.5), 30.0)
        self.assertIsNone(dashboard_percentile([], 0.95))


if __name__ == "__main__":
    unittest.main()
