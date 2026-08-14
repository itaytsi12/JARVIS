from brain.agent import run_agent
from brain.router import route_command
from voice.response_formatter import format_spoken_response

cmds = [
    'open youtube',
    'search youtube for veritasium',
    'open notepad and type hello',
    'volume down',
    'take a screenshot',
]

for cmd in cmds:
    print('CMD:', cmd)
    route = route_command(cmd)
    resp = run_agent(cmd)
    print('TERMINAL:', resp)
    print('SPOKEN:', format_spoken_response(cmd, route, resp))
    print('---')
