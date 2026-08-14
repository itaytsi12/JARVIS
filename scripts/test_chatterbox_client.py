import importlib,traceback
m=importlib.import_module('voice.tts.chatterbox_tts')
try:
    m.speak('Okay, opening YouTube.', lang='en')
    print('speak call returned')
except Exception as e:
    traceback.print_exc()
    print('speak failed', e)
