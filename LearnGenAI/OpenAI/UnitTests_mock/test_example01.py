import unittest
from unittest.mock import patch
from LearnGenAI.OpenAI.UnitTests_mock.codes_testing01 import charge_card

class TestChargeCard(unittest.TestCase):
    @patch('codes_testing01.requests.post')
    def test_charge_card_success(self, mock_post):
        # Mock the response from the payment gateway
        mock_post.return_value.json.return_value = {
            "status": "success",
            "transaction_id": "12345"
        }
        
        result = charge_card("12345", 100)
        print(result)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["transaction_id"], "12345")

    @patch('codes_testing01.requests.post')
    def test_charge_card_failure(self, mock_post):
        # Mock the response from the payment gateway for a failed transaction
        mock_post.return_value.json.return_value = {
            "status": "failure",
            "error_message": "Card declined"
        }
        
        result = charge_card("12345", 100)
        print(result)
        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["error_message"], "Card declined")

if __name__ == '__main__':
    unittest.main()