import base64
import io
import os
import time

from openai import OpenAI


# `.env` is loaded exactly once, from the project root, by
# `config/settings.py` -- importing it here is what guarantees that has
# happened before the environment is read below. A local `load_dotenv()`
# call would search upward from the CURRENT WORKING DIRECTORY instead,
# which is how the runtime silently ended up with no API key when it was
# started from anywhere but the repository root.
import config  # noqa: F401  -- imported for its .env loading side effect

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def _bounded_image_payload(image_path):
    raw=open(image_path,"rb").read();dimensions=None;max_dimension=max(640,int(os.getenv("JARVIS_VISION_MAX_DIMENSION","1600")))
    try:
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as image:
            dimensions=image.size
            if max(image.size)>max_dimension:
                image.thumbnail((max_dimension,max_dimension));output=io.BytesIO();image.save(output,format="PNG",optimize=True);raw=output.getvalue();dimensions=image.size
    except Exception:
        pass
    return base64.b64encode(raw).decode("utf-8"),len(raw),dimensions


def analyze_screen(image_path: str, question: str) -> dict:
    image_base64,image_bytes,image_dimensions=_bounded_image_payload(image_path)

    model=os.getenv("JARVIS_VISION_MODEL","gpt-5-mini");started=time.perf_counter()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": question
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_base64}"
                    }
                ]
            }
        ]
    )

    usage=getattr(response,"usage",None);answer=response.output_text
    return {"success":bool(answer),"message":answer or "I couldn't analyze the screen.","answer":answer,"model":model,"model_calls":1,"input_tokens":getattr(usage,"input_tokens",0) or 0,"output_tokens":getattr(usage,"output_tokens",0) or 0,"latency_ms":(time.perf_counter()-started)*1000,"image_input_bytes":image_bytes,"image_dimensions":image_dimensions,"screenshot_path":str(image_path),"error":None if answer else "empty_vision_response"}
