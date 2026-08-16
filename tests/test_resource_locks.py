import threading,time,unittest
from unittest.mock import patch

from brain.executor import Executor
from brain.agent_runtime import AgentRuntime
from brain.models import Action,Plan,ToolResult
from brain.resource_locks import WAIT_OBJECT_0,_acquire_named_mutex,_release_named_mutex,acquire_action_resource,acquire_named_resource,resource_for_tool
from brain.task_supervisor import CancellationToken


class ResourceLockTests(unittest.TestCase):
    def test_windows_named_mutex_is_acquired_and_released(self):
        class Kernel:
            def __init__(self):self.released=[];self.closed=[]
            def CreateMutexW(self,*args):self.name=args[-1];return 7
            def WaitForSingleObject(self,*_):return WAIT_OBJECT_0
            def ReleaseMutex(self,handle):self.released.append(handle)
            def CloseHandle(self,handle):self.closed.append(handle)
        kernel=Kernel()
        with patch("brain.resource_locks.os.name","nt"):
            handle=_acquire_named_mutex("desktop_input",None,time.perf_counter()+1,kernel)
            _release_named_mutex(handle,kernel)
        self.assertIn("desktop_input",kernel.name);self.assertEqual(kernel.released,[7]);self.assertEqual(kernel.closed,[7])

    def test_deterministic_website_open_shares_browser_resource(self):
        self.assertEqual(resource_for_tool("open_website"),"browser_session")
    def test_whole_action_plans_do_not_interleave_context_mutations(self):
        order=[];started=threading.Barrier(2)
        class Runtime(AgentRuntime):
            def _execute_action(self,action,cancellation_token=None):
                order.append(action.args["owner"]);time.sleep(.02);return ToolResult(True,action.tool,"ok")
        def run(owner):
            runtime=Runtime(trace=False);started.wait();runtime.execute(Plan(owner,[Action("step",{"owner":owner}),Action("step",{"owner":owner})]))
        threads=[threading.Thread(target=run,args=(owner,)) for owner in ("A","B")]
        for thread in threads:thread.start()
        for thread in threads:thread.join(2)
        self.assertIn(order,[["A","A","B","B"],["B","B","A","A"]])
    def test_desktop_input_is_exclusive_across_threads(self):
        active=0;maximum=0;guard=threading.Lock()
        def operation():
            nonlocal active,maximum
            with acquire_action_resource("type_text"):
                with guard:active+=1;maximum=max(maximum,active)
                time.sleep(.03)
                with guard:active-=1
        threads=[threading.Thread(target=operation) for _ in range(3)]
        for thread in threads:thread.start()
        for thread in threads:thread.join()
        self.assertEqual(maximum,1)
    def test_repository_resources_are_path_scoped(self):
        active=0;maximum=0;guard=threading.Lock()
        def operation():
            nonlocal active,maximum
            with acquire_named_resource("repository:c:/same"):
                with guard:active+=1;maximum=max(maximum,active)
                time.sleep(.02)
                with guard:active-=1
        threads=[threading.Thread(target=operation) for _ in range(2)]
        for thread in threads:thread.start()
        for thread in threads:thread.join()
        self.assertEqual(maximum,1)

    def test_read_only_tools_do_not_take_desktop_lock(self):
        self.assertIsNone(resource_for_tool("get_time"));self.assertEqual(resource_for_tool("browser_click"),"browser_session");self.assertEqual(resource_for_tool("speak_response"),"speaker")

    def test_cancelled_wait_does_not_execute(self):
        token=CancellationToken();entered=threading.Event();finished=threading.Event()
        def waiter():
            try:
                with acquire_action_resource("type_text",token):entered.set()
            except RuntimeError:finished.set()
        with acquire_action_resource("type_text"):
            thread=threading.Thread(target=waiter);thread.start();time.sleep(.06);token.cancel()
        thread.join(1);self.assertFalse(entered.is_set());self.assertTrue(finished.is_set())


if __name__=="__main__":unittest.main()
