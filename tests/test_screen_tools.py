import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from PIL import Image

from vision import screenshot,screen_analyzer
from brain import tool_router


class ScreenToolTests(unittest.TestCase):
    def test_screenshot_returns_verified_structured_result(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(screenshot,"SCREENSHOT_DIR",Path(folder)),patch.object(screenshot.ImageGrab,"grab") as grab:
            grab.return_value.save.side_effect=lambda path:Path(path).write_bytes(b"png")
            result=screenshot.take_screenshot()
        self.assertTrue(result["success"]);self.assertTrue(result["verified"]);self.assertGreater(result["bytes"],0)

    def test_screen_analysis_reports_model_usage_without_embedding_image(self):
        response=SimpleNamespace(output_text="Notepad is open.",usage=SimpleNamespace(input_tokens=20,output_tokens=5))
        with tempfile.TemporaryDirectory() as folder:
            image=Path(folder)/"screen.png";image.write_bytes(b"image")
            with patch.object(screen_analyzer.client.responses,"create",return_value=response):result=screen_analyzer.analyze_screen(str(image),"What is open?")
        self.assertTrue(result["success"]);self.assertEqual(result["model_calls"],1);self.assertEqual(result["input_tokens"],20)
        self.assertNotIn("data:image",str(result));self.assertNotIn("base64",str(result))

    def test_analysis_capture_is_unique_and_removed_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            image=Path(folder)/"analysis.png";image.write_bytes(b"png")
            with patch.object(tool_router,"take_screenshot",return_value={"success":True,"path":str(image)}),patch.object(tool_router,"analyze_screen",return_value={"success":True,"message":"ok"}):
                result=tool_router.execute_tool("analyze_screen",{"question":"What is open?"})
            self.assertTrue(result["success"]);self.assertFalse(result["screenshot_retained"]);self.assertFalse(image.exists())

    def test_rapid_explicit_screenshots_do_not_reuse_filename(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(screenshot,"SCREENSHOT_DIR",Path(folder)),patch.object(screenshot.ImageGrab,"grab") as grab:
            grab.return_value.save.side_effect=lambda path:Path(path).write_bytes(b"png")
            first=screenshot.take_screenshot();second=screenshot.take_screenshot()
        self.assertNotEqual(first["path"],second["path"])

    def test_large_vision_input_is_downscaled_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as folder:
            image=Path(folder)/"large.png";Image.new("RGB",(3200,1800),"white").save(image);original_size=Image.open(image).size
            payload,size,dimensions=screen_analyzer._bounded_image_payload(image)
        self.assertTrue(payload);self.assertGreater(size,0);self.assertEqual(dimensions,(1600,900));self.assertEqual(original_size,(3200,1800))


if __name__=="__main__":unittest.main()
