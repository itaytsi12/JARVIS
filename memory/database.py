from __future__ import annotations
import sqlite3, threading
from pathlib import Path

class MemoryDatabase:
    def __init__(self, path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self._lock=threading.RLock(); self.connection=sqlite3.connect(self.path,check_same_thread=False)
        self.connection.row_factory=sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL"); self.connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()
    def _migrate(self):
        with self.connection: self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
        INSERT INTO schema_version SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM schema_version);
        CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,started_at TEXT NOT NULL,ended_at TEXT,summary TEXT,metadata_json TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,goal TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,started_at TEXT,completed_at TEXT,workspace TEXT,repository TEXT,branch TEXT,allowed_paths_json TEXT NOT NULL DEFAULT '[]',forbidden_paths_json TEXT NOT NULL DEFAULT '[]',allowed_commands_json TEXT NOT NULL DEFAULT '[]',max_runtime REAL NOT NULL DEFAULT 3600,max_iterations INTEGER NOT NULL DEFAULT 20,success_criteria TEXT,current_iteration INTEGER NOT NULL DEFAULT 0,current_hypothesis TEXT,last_result TEXT,blocked_reason TEXT,requires_approval INTEGER NOT NULL DEFAULT 0,checkpoint_json TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS entities(id TEXT PRIMARY KEY,session_id TEXT,type TEXT NOT NULL,name TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'active',created_at TEXT NOT NULL,updated_at TEXT NOT NULL,metadata_json TEXT NOT NULL DEFAULT '{}',FOREIGN KEY(session_id) REFERENCES sessions(id));
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,task_id TEXT,entity_id TEXT,kind TEXT NOT NULL,message TEXT NOT NULL,created_at TEXT NOT NULL,metadata_json TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS relations(id INTEGER PRIMARY KEY AUTOINCREMENT,source_id TEXT NOT NULL,relation TEXT NOT NULL,target_id TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(source_id,relation,target_id));
        CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY AUTOINCREMENT,kind TEXT NOT NULL,key TEXT,text TEXT NOT NULL,importance INTEGER NOT NULL DEFAULT 1,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,metadata_json TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY,task_id TEXT,path TEXT,kind TEXT NOT NULL,size_bytes INTEGER NOT NULL DEFAULT 0,disposable INTEGER NOT NULL DEFAULT 1,archived_uri TEXT,created_at TEXT NOT NULL,metadata_json TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS summaries(id INTEGER PRIMARY KEY AUTOINCREMENT,subject_type TEXT NOT NULL,subject_id TEXT NOT NULL,text TEXT NOT NULL,created_at TEXT NOT NULL,metadata_json TEXT NOT NULL DEFAULT '{}');
        CREATE INDEX IF NOT EXISTS idx_entities_recent ON entities(type,updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_events_recent ON events(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status,updated_at DESC);
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_search USING fts5(kind,reference_id UNINDEXED,text);
        """)
    def execute(self,sql,parameters=()):
        with self._lock,self.connection: return self.connection.execute(sql,parameters)
    def query(self,sql,parameters=()):
        with self._lock: return list(self.connection.execute(sql,parameters))
    def close(self):
        with self._lock: self.connection.close()
