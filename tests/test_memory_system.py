import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from memory import LocalFilesystemArchive, MemoryManager, redact
from brain.models import Action, ToolResult
from brain import agent

class MemorySystemTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.memory=MemoryManager(self.root/"memory.db")
    def tearDown(self): self.memory.db.close(); self.temp.cleanup()
    def test_global_agent_memory_is_isolated_during_pytest(self):
        self.assertNotEqual(agent.memory_manager.db.path.resolve(),(Path.cwd()/"data"/"jarvis_memory.sqlite3").resolve())
        self.assertTrue(agent.memory_manager.db.path.parent.name.startswith("jarvis-memory-pytest-"))
    def test_persistence_reference_ambiguity_and_stale_entities(self):
        session=self.memory.start_session(); first=self.memory.remember_entity("application","notepad",session,pid=1,hwnd=2)
        self.assertEqual(self.memory.resolve("the Notepad you opened",session).entity["id"],first["id"])
        self.memory.remember_entity("application","notepad",session,pid=3,hwnd=4)
        self.assertEqual(self.memory.resolve("the Notepad you opened",session).status,"ambiguous")
        self.memory.update_entity(first["id"],status="stale")
        self.memory.db.close(); self.memory=MemoryManager(self.root/"memory.db")
        self.assertEqual(self.memory.resolve("the Notepad you opened",session).status,"resolved")
    def test_search_file_search_and_secret_redaction(self):
        session=self.memory.start_session(); self.memory.remember_entity("file","hello.txt",session,path="C:/Desktop/hello.txt")
        self.memory.remember_entity("search","Jude Law",session,provider="youtube")
        self.memory.record_event("note","password is hunter2 and API key=abc")
        self.assertTrue(self.memory.search("Jude")); rows=self.memory.db.query("SELECT message FROM events")
        self.assertNotIn("hunter2",rows[0][0]); self.assertNotIn("abc",rows[0][0])
        raw="ghp_abcdefghijklmnop Bearer abc.def.ghi Cookie: sid=private"
        safe=redact(raw);self.assertNotIn("abcdefghijklmnop",safe);self.assertNotIn("abc.def.ghi",safe);self.assertNotIn("sid=private",safe)
    def test_artifact_cleanup_archive_and_budget(self):
        archive=LocalFilesystemArchive(self.root/"archive"); self.memory.archive=archive; p=self.root/"raw.wav"; p.write_bytes(b"x"*100)
        aid=self.memory.add_artifact(p); old=(datetime.now(timezone.utc)-timedelta(days=8)).isoformat(); self.memory.db.execute("UPDATE artifacts SET created_at=? WHERE id=?",(old,aid))
        result=self.memory.compact(); self.assertFalse(p.exists()); self.assertTrue(archive.exists(aid)); self.assertGreaterEqual(result["removed_bytes"],100)
    def test_obsidian_disabled_and_enabled(self):
        with patch.dict(os.environ,{"OBSIDIAN_MEMORY_ENABLED":"false"}): self.assertEqual(self.memory.export_obsidian(self.root/"vault"),[])
        now=datetime.now(timezone.utc).isoformat(); self.memory.db.execute("INSERT INTO tasks(id,goal,status,created_at,updated_at) VALUES(?,?,?,?,?)",("t","Goal","PAUSED",now,now))
        with patch.dict(os.environ,{"OBSIDIAN_MEMORY_ENABLED":"true"}): paths=self.memory.export_obsidian(self.root/"vault")
        self.assertEqual(len(paths),1); self.assertIn("Status: PAUSED",paths[0].read_text())
    def test_budget_prunes_only_disposable_artifacts(self):
        disposable=self.root/"temp.bin"; important=self.root/"important.bin"; disposable.write_bytes(b"x"*100); important.write_bytes(b"y"*100)
        self.memory.add_artifact(disposable,disposable=True); self.memory.add_artifact(important,disposable=False); self.memory.max_local_mb=0; self.memory.compact()
        self.assertFalse(disposable.exists()); self.assertTrue(important.exists())
    def test_simple_fast_path_action_updates_memory(self):
        with patch.object(agent,"memory_manager",self.memory), patch.object(agent.agent_runtime,"session_id",self.memory.start_session()):
            agent._remember_action(Action("open_application",{"app_name":"notepad"}),ToolResult(True,"open_application",data={"pid":123}))
            agent._remember_action(Action("open_website",{"url":"https://www.youtube.com/results?search_query=Jude+Law"}),ToolResult(True,"open_website"))
        self.assertEqual(self.memory.resolve("the Notepad you opened").status,"resolved"); self.assertEqual(self.memory.resolve("that YouTube search").entity["name"],"Jude Law")

if __name__=="__main__": unittest.main()
