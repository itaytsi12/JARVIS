import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

from training_data.recorder import DatasetRecorder
from voice.speech_journal import begin_speech,finish_speech


class SpeechJournalTests(unittest.TestCase):
    def test_speech_records_provider_and_length_without_content(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder=DatasetRecorder(Path(folder)/"speech.db",async_writes=False)
            try:
                with patch("voice.speech_journal.get_recorder",return_value=recorder):
                    journal=begin_speech("en",42,"parent");self.assertTrue(finish_speech(journal,{"success":True,"provider":"openai","attempted_providers":["chatterbox","openai"],"fallback_from":["chatterbox"],"spoken_chars":42}))
                raw=" ".join(row[0] for row in recorder._connection.execute("SELECT payload_json FROM raw_events"))
                self.assertIn('"provider":"openai"',raw);self.assertIn('"fallback_from":["chatterbox"]',raw);self.assertIn('"text_length":42',raw);self.assertNotIn("secret spoken text",raw)
                self.assertEqual(recorder.stats()["verified_records"],1)
            finally:recorder.close()


if __name__=="__main__":unittest.main()
